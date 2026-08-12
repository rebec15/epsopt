"""
portfolio_opt.py – Bi-criterial portfolio optimization with two scenarios of probability p1,p2

Graph:
    graph F = { (x,y) ∈ R^n × R^2 : ∃ z ∈ R^n :
                y[0] ≤ p1* r1^T z1 + p2* r2^T z2            (expected yield)
                y[1] ≤ -p1* z1^T Q1 z1 - p2* z2^T Q2 z2     (expected negative risk)
                z1 ≥ 0,  sum(z1) = 1
                x ≥ 0,  sum(x) = 1
                ||x - z1||_∞ ≤ τ 
                z2 ≥ 0,  sum(z2) = 1
                x ≥ 0,  sum(x) = 1
                ||x - z2||_∞ ≤ τ 
                }

x - reference portfolio stage 1; F(x) are all (yield, -risk)-pairs achievable with portfolio z1,z2 under scenario 1,2 in stage 2 under condition that z1,z2 is within a τ-ball around reference portfolio x.
"""

from __future__ import annotations
from typing import Callable, List, Optional, cast
import csv
import math
from datetime import datetime
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial import ConvexHull

import os

from epsopt.graph import Graph
from epsopt.csop import CSOP
from epsopt.approximation import Approximation
from epsopt.convex_set import ConvexSet
from epsopt.convex_set_approximator import ConvexSetApproximator


class PortfolioOptScen:
    """
    Parameters
    r : np.ndarray, shape (n,)
        yields of the n given assets in different scenarios.
    Q : np.ndarray, shape (n, n)
        covariance matrix (positive semidefinite) in different scenarios.
    tau : float
        radius for "maximum re-structuring": ||x - z||_∞ ≤ τ.
    epsilon : float
        tolerance ε for gammaEps-optimizer algorithm.
    solver : str, optional
        cvxpy-Solver.
    """

    # instance attributes
    graph: Graph
    csop: CSOP
    p1: float
    p2: float
    r1: np.ndarray
    Q1: np.ndarray
    r2: np.ndarray
    Q2: np.ndarray
    tau: float
    epsilon: float
    solver: Optional[str]
    _plot_viewport: Optional[tuple[float, float, float, float]]

    def __init__(
        self,
        r1: np.ndarray,
        Q1: np.ndarray,
        p1: float,
        r2: np.ndarray,
        Q2: np.ndarray,
        p2: float,
        tau: float,
        epsilon: float = 1e-2,
        *,
        solver: Optional[str] = None,
    ) -> None:
        
        # Input format check
        r1 = np.asarray(r1, dtype=float)
        r2 = np.asarray(r2, dtype=float)
        Q1 = np.asarray(Q1, dtype=float)
        Q2 = np.asarray(Q2, dtype=float)

        if r1.ndim != 1:
            raise ValueError(f"r1 must be a 1D array, got shape {r1.shape}.")
        if r2.ndim != 1:
            raise ValueError(f"r2 must be a 1D array, got shape {r2.shape}.")

        n = r1.shape[0]
        if r2.shape[0] != n:
            raise ValueError(
                f"len(r1) and len(r2) must match, got len(r1)={n} and len(r2)={r2.shape[0]}."
            )
        if Q1.shape != (n, n):
            raise ValueError(
                f"Q1 must have shape ({n}, {n}), got {Q1.shape}."
            )
        if Q2.shape != (n, n):
            raise ValueError(
                f"Q2 must have shape ({n}, {n}), got {Q2.shape}."
            )

        if p1 < 0 or p2 < 0:
            raise ValueError(f"Probabilities p1 and p2 must be >= 0, got p1={p1}, p2={p2}.")
        if not math.isclose(p1 + p2, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"Sum of probabilities of scenarios p1 + p2 must equal 1, got p1+p2={p1 + p2}.")

        eigvals1 = np.linalg.eigvalsh(Q1)
        eigvals2 = np.linalg.eigvalsh(Q2)
        if np.any(eigvals1 < -1e-10):
            raise ValueError(
                f"Q1 must be positive semidefinite, but has negative eigenvalue(s): {eigvals1[eigvals1 < 0]}."
            )
        if np.any(eigvals2 < -1e-10):
            raise ValueError(
                f"Q2 must be positive semidefinite, but has negative eigenvalue(s): {eigvals2[eigvals2 < 0]}."
            )
        
        self.r1 = r1
        self.Q1 = Q1
        self.p1 = p1
        self.r2 = r2
        self.Q2 = Q2
        self.p2 = p2
        self.tau = tau
        self.epsilon = epsilon
        self.solver = solver
        self._plot_viewport = None

        # n: reference portfolio x ∈ R^n
        # q=2: image space (yield, -risk)
        # m=n: shadow variable z ∈ R^n
        # y = (yield, -risk) is only upper-bounded by constraints; downward directions
        # are recession directions, so use K = cone{(-1,0), (0,-1)}.
        recc_generators = [
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
        graph = Graph(
            n=n,
            q=2,
            m=2*n,
            name="PortfolioGraph",
            recession_cone_generators=recc_generators,
        )

        graph.add_constraint_fn(lambda x, y, z: [
            y[0] <= p1* (r1 @ z[0:n]) + p2* (r2 @ z[n:2*n]) ,                       # yield 
            y[1] <= -p1* cp.quad_form(z[0:n], Q1) -p2* cp.quad_form(z[n:2*n], Q2),  # -risk
            z >= 0,                                                                 # non-negativity
            cp.sum(z[0:n]) == 1,              
            cp.sum(z[n:2*n]) == 1,                                                  # full investment
            cp.norm(x - z[0:n], "inf") <= tau,
            cp.norm(x - z[n:2*n], "inf") <= tau,                                    # bound on re-allocation amount
            x >= 0,
            x <= 1
        ])

        self.graph = graph
        self.csop = CSOP(graph, solver=solver)

    def run(self, y0: Optional[np.ndarray] = None, verbose: bool = False,
            compute_gamma_upper_bound: bool = False,
            on_iteration: Optional[Callable[[int, Approximation], None]] = None) -> dict:
        """run gammaEps-optimizer algorithm

        Parameters
        y0 : np.ndarray, shape (2,), optional
            Starting point in image space = (yield, -risk).
            If None, a feasible point is selected automatically from img F.
        verbose : bool
        compute_gamma_upper_bound : bool
            If True, compute and return/display the upper bound for
            ``gamma``.
        on_iteration : callable (int, Approximation) -> None, optional
            Called with (0, initial_approx) before the loop and
            (i, approx_snapshot) after each algorithm step.
            Collect these to visualise the evolution with plot_iterations().

        Returns
        dict with keys: "y_choice", "x", "eps", "approx"
        """

        result = self.csop.computeEpsOptimizer(
            y=y0,
            eps=self.epsilon,
            compute_gamma_upper_bound=compute_gamma_upper_bound,
            on_iteration=on_iteration,
        )

        if verbose:
            gamma_val = result.get("gamma_upper_bound")
            print(f"starting point y0 = (yield,-risk)   = {result['y_choice']}")
            print(f"portfolio x*    = {result['x']}")
            if gamma_val is None:
                print("gamma upper bound = n/a")
            elif np.isinf(gamma_val):
                print("gamma upper bound = inf")
            else:
                print(f"gamma upper bound = {float(gamma_val):.6g}")
            print(f"approximation   = {result['approx']}")

        return result

    @staticmethod
    def _sort_vertices_clockwise(vertices: np.ndarray) -> np.ndarray:
        """Return vertices sorted clockwise around their centroid."""
        verts = np.asarray(vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) <= 1:
            return verts

        center = verts.mean(axis=0)
        angles = np.arctan2(verts[:, 1] - center[1], verts[:, 0] - center[0])
        # Descending angle order gives clockwise ordering.
        order = np.argsort(-angles)
        return verts[order]

    @staticmethod
    def _plot_recession_rays_2d(
        ax: Axes,
        anchors: np.ndarray,
        vertices: np.ndarray,
        generators: np.ndarray,
        *,
        color: str,
        label: Optional[str] = None,
    ) -> None:
        """Draw 2D recession rays for unbounded sets of form P + cone(generators)."""
        if generators.size == 0:
            return
        if generators.ndim != 2 or generators.shape[1] != 2:
            return
        norms = np.linalg.norm(generators, axis=1)
        keep = norms > 1e-12
        if not np.any(keep):
            return
        rays = generators[keep] / norms[keep][:, np.newaxis]
        if anchors.size == 0:
            return

        span = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-8)
        ray_len = 0.35 * float(np.linalg.norm(span))
        first = True
        for a in anchors:
            for r in rays:
                ax.arrow(
                    float(a[0]),
                    float(a[1]),
                    float(ray_len * r[0]),
                    float(ray_len * r[1]),
                    width=0.0,
                    head_width=0.02 * float(max(span[0], span[1])),
                    length_includes_head=True,
                    color=color,
                    alpha=0.7,
                    linestyle="--",
                    zorder=4,
                    label=label if first else None,
                )
                first = False

    @staticmethod
    def _normalize_rows(a: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        if a.size == 0:
            return a
        norms = np.linalg.norm(a, axis=1)
        keep = norms > tol
        if not np.any(keep):
            return np.empty((0, a.shape[1]), dtype=float)
        out = a[keep].copy()
        out /= norms[keep][:, np.newaxis]
        return out

    @staticmethod
    def _deduplicate_rows(a: np.ndarray, decimals: int = 12) -> np.ndarray:
        if a.size == 0:
            return a
        rounded = np.round(a, decimals=decimals)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        return a[np.sort(idx)]

    @staticmethod
    def _deduplicate_vertices_for_plot(vertices: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Remove near-identical 2D vertices."""
        V = np.asarray(vertices, dtype=float)
        if V.ndim != 2 or V.shape[1] != 2 or len(V) <= 1:
            return V

        keep_idx: List[int] = []
        for i, p in enumerate(V):
            if not keep_idx:
                keep_idx.append(i)
                continue
            kept = V[keep_idx]
            if float(np.min(np.linalg.norm(kept - p, axis=1))) > tol:
                keep_idx.append(i)
        return V[np.asarray(keep_idx, dtype=int)]

    def _solve_problem(self, problem: cp.Problem, *, warm_start: bool = False) -> None:
        """Solve a CVXPY problem with the configured solver (or CVXPY default)."""
        if self.solver is None:
            problem.solve(warm_start=warm_start)
        else:
            solver_name = str(self.solver).upper()
            if solver_name == "SCS":
                problem.solve(
                    solver=self.solver,
                    warm_start=warm_start,
                    eps=1e-7,
                    max_iters=200000,
                    acceleration_lookback=50,
                )
            else:
                problem.solve(solver=self.solver, warm_start=warm_start)

    def _prune_vertices_for_plot(
        self,
        vertices: np.ndarray,
        generators: np.ndarray,
        *,
        tol: float = 1e-7,
    ) -> np.ndarray:
        """Drop numerically redundant points in conv(vertices)+cone(generators)."""
        V = self._deduplicate_vertices_for_plot(np.asarray(vertices, dtype=float), tol=tol)
        if V.ndim != 2 or V.shape[1] != 2 or len(V) <= 1:
            return V

        G = np.asarray(generators, dtype=float)
        if G.size == 0:
            G = np.empty((0, 2), dtype=float)
        elif G.ndim != 2 or G.shape[1] != 2:
            raise ValueError("generators must have shape (r,2).")

        keep = np.ones(len(V), dtype=bool)
        for i in range(len(V)):
            others = np.delete(V, i, axis=0)
            if len(others) == 0:
                continue

            lam = cp.Variable(len(others), nonneg=True)
            constraints = cast(List[cp.Constraint], [cp.sum(lam) == 1])
            image_point = others.T @ lam

            if G.size > 0:
                mu = cp.Variable(G.shape[0], nonneg=True)
                image_point = image_point + G.T @ mu

            resid = cp.norm(image_point - V[i], 2)
            prob = cp.Problem(cp.Minimize(resid), constraints)
            try:
                self._solve_problem(prob, warm_start=True)
            except Exception:
                continue

            if prob.value is None:
                continue

            resid_val = float(np.asarray(prob.value).item())
            if np.isfinite(resid_val) and resid_val <= 25.0 * tol:
                keep[i] = False

        out = V[keep]
        if len(out) == 0:
            return V[[0]]
        return out

    @staticmethod
    def _compute_viewport_2d(
        vertices: np.ndarray,
        generators: np.ndarray,
        *,
        pad_ratio: float = 0.08,
        ray_scale: float = 2.0,
    ) -> tuple[float, float, float, float]:
        """Compute a plotting window for conv(vertices)+cone(generators)."""
        V = np.asarray(vertices, dtype=float)
        if V.ndim != 2 or V.shape[1] != 2 or len(V) == 0:
            return (-1.0, 1.0, -1.0, 1.0)

        G = np.asarray(generators, dtype=float)
        if G.size == 0:
            G = np.empty((0, 2), dtype=float)

        span = np.maximum(V.max(axis=0) - V.min(axis=0), 1e-6)
        base = float(np.linalg.norm(span))
        ray_len = ray_scale * base

        ext_pts = [V]
        if G.size > 0:
            norms = np.linalg.norm(G, axis=1)
            keep = norms > 1e-12
            if np.any(keep):
                rays = G[keep] / norms[keep][:, np.newaxis]
                ext_pts.append(V[:, None, :] + ray_len * rays[None, :, :])

        all_pts = np.vstack([p.reshape(-1, 2) for p in ext_pts])
        mins = np.min(all_pts, axis=0)
        maxs = np.max(all_pts, axis=0)
        pad = pad_ratio * float(max(maxs - mins))
        x_min, y_min = mins - pad
        x_max, y_max = maxs + pad
        return (float(x_min), float(x_max), float(y_min), float(y_max))

    @staticmethod
    def _expand_viewport_with_vertices(
        viewport: tuple[float, float, float, float],
        vertices_list: List[np.ndarray],
        *,
        pad_ratio: float = 0.02,
    ) -> tuple[float, float, float, float]:
        """Expand viewport so that all provided 2D vertices are inside."""
        x_min, x_max, y_min, y_max = viewport
        valid_sets: List[np.ndarray] = []

        for verts in vertices_list:
            arr = np.asarray(verts, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                valid_sets.append(arr)

        if not valid_sets:
            return viewport

        all_pts = np.vstack(valid_sets)
        mins = np.min(all_pts, axis=0)
        maxs = np.max(all_pts, axis=0)

        x_min = min(float(x_min), float(mins[0]))
        x_max = max(float(x_max), float(maxs[0]))
        y_min = min(float(y_min), float(mins[1]))
        y_max = max(float(y_max), float(maxs[1]))

        span = max(x_max - x_min, y_max - y_min, 1e-9)
        pad = pad_ratio * span
        return (x_min - pad, x_max + pad, y_min - pad, y_max + pad)

    @staticmethod
    def _square_viewport(
        viewport: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Return a viewport with equal x/y span, centered at original viewport center."""
        x_min, x_max, y_min, y_max = viewport
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        span = max(x_max - x_min, y_max - y_min, 1e-9)
        half = 0.5 * span
        return (cx - half, cx + half, cy - half, cy + half)

    @classmethod
    def _compute_cone_facet_normals_2d(cls, generators: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        """Compute facet normals of a 2D cone from its generators."""
        G = np.asarray(generators, dtype=float)
        if G.size == 0:
            return np.empty((0, 2), dtype=float)
        if G.ndim != 2 or G.shape[1] != 2:
            return np.empty((0, 2), dtype=float)

        norms = np.linalg.norm(G, axis=1)
        keep = norms > tol
        if not np.any(keep):
            return np.empty((0, 2), dtype=float)

        rays = G[keep] / norms[keep][:, np.newaxis]
        points = np.vstack([np.zeros((1, 2)), rays])
        try:
            hull = ConvexHull(points)
        except Exception:
            return np.empty((0, 2), dtype=float)

        def compatible(n: np.ndarray) -> bool:
            return bool(np.all(G @ n <= tol))

        candidates: List[np.ndarray] = []
        for row in hull.equations:
            n = np.asarray(row[:-1], dtype=float)
            d = float(row[-1])
            if abs(d) > 1e-7:
                continue
            if compatible(n):
                candidates.append(n)
            elif compatible(-n):
                candidates.append(-n)

        if not candidates:
            return np.empty((0, 2), dtype=float)

        arr = np.asarray(candidates, dtype=float)
        return cls._deduplicate_rows(cls._normalize_rows(arr))

    @classmethod
    def _build_unbounded_set_inequalities_2d(
        cls,
        vertices: np.ndarray,
        generators: np.ndarray,
        tol: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A,b,is_cone) for conv(vertices)+cone(generators): A y <= b."""
        V = np.asarray(vertices, dtype=float)
        if V.ndim != 2 or V.shape[1] != 2 or len(V) == 0:
            raise ValueError("vertices must have shape (k,2) with k>=1.")

        G = np.asarray(generators, dtype=float)
        if G.size == 0:
            G = np.empty((0, 2), dtype=float)
        elif G.ndim != 2 or G.shape[1] != 2:
            raise ValueError("generators must have shape (r,2).")

        def compatible(n: np.ndarray) -> bool:
            if G.size == 0:
                return True
            return bool(np.all(G @ n <= tol))

        normals: List[np.ndarray] = []
        from_cone: List[bool] = []
        if len(V) >= 3:
            hull = ConvexHull(V)
            for n in hull.equations[:, :-1]:
                n_arr = np.asarray(n, dtype=float)
                if compatible(n_arr):
                    normals.append(n_arr)
                    from_cone.append(False)
        elif len(V) == 2:
            t = V[1] - V[0]
            if np.linalg.norm(t) > tol:
                cands = [np.array([t[1], -t[0]]), np.array([-t[1], t[0]])]
                for n in cands:
                    if compatible(n):
                        normals.append(n)
                        from_cone.append(False)

        cone_normals = cls._compute_cone_facet_normals_2d(G, tol=tol)
        if cone_normals.size > 0:
            normals.extend([cone_normals[i, :] for i in range(cone_normals.shape[0])])
            from_cone.extend([True] * cone_normals.shape[0])

        if not normals:
            raise RuntimeError("Could not derive facet normals for unbounded set drawing.")

        raw = cls._normalize_rows(np.asarray(normals, dtype=float))
        tags = np.asarray(from_cone, dtype=bool)
        unique_rows: List[np.ndarray] = []
        unique_tags: List[bool] = []
        seen: dict[tuple[float, float], int] = {}
        for i in range(raw.shape[0]):
            key = tuple(np.round(raw[i], decimals=12))
            if key not in seen:
                seen[key] = len(unique_rows)
                unique_rows.append(raw[i])
                unique_tags.append(bool(tags[i]))
            else:
                j = seen[key]
                unique_tags[j] = bool(unique_tags[j] or tags[i])

        A = np.asarray(unique_rows, dtype=float)
        is_cone = np.asarray(unique_tags, dtype=bool)
        b = np.max(V @ A.T, axis=0)
        return A, b, is_cone

    @classmethod
    def _draw_unbounded_boundary_2d(
        cls,
        ax: Axes,
        vertices: np.ndarray,
        generators: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        is_cone: np.ndarray,
        *,
        edge_color: str,
        viewport: Optional[tuple[float, float, float, float]] = None,
        tol: float = 1e-7,
    ) -> None:
        """Draw only true outer boundary parts (finite edges + recession rays)."""
        V = np.asarray(vertices, dtype=float)
        if V.ndim != 2 or V.shape[1] != 2 or len(V) == 0:
            return
        G = np.asarray(generators, dtype=float)
        if G.size == 0:
            G = np.empty((0, 2), dtype=float)

        span = np.maximum(V.max(axis=0) - V.min(axis=0), 1e-6)

        if viewport is None:
            viewport = cls._compute_viewport_2d(V, G)
        x_min, x_max, y_min, y_max = viewport

        def _ray_endpoint_on_viewport(anchor: np.ndarray, direction: np.ndarray) -> Optional[np.ndarray]:
            dx = float(direction[0])
            dy = float(direction[1])
            x0 = float(anchor[0])
            y0 = float(anchor[1])
            ts: List[float] = []

            if abs(dx) > 1e-14:
                t_xmin = (x_min - x0) / dx
                y_at_xmin = y0 + t_xmin * dy
                if t_xmin > tol and y_min - tol <= y_at_xmin <= y_max + tol:
                    ts.append(t_xmin)

                t_xmax = (x_max - x0) / dx
                y_at_xmax = y0 + t_xmax * dy
                if t_xmax > tol and y_min - tol <= y_at_xmax <= y_max + tol:
                    ts.append(t_xmax)

            if abs(dy) > 1e-14:
                t_ymin = (y_min - y0) / dy
                x_at_ymin = x0 + t_ymin * dx
                if t_ymin > tol and x_min - tol <= x_at_ymin <= x_max + tol:
                    ts.append(t_ymin)

                t_ymax = (y_max - y0) / dy
                x_at_ymax = x0 + t_ymax * dx
                if t_ymax > tol and x_min - tol <= x_at_ymax <= x_max + tol:
                    ts.append(t_ymax)

            if not ts:
                return None

            t_end = min(ts)
            return anchor + t_end * direction

        # Finite outer edges induced by non-cone facet normals.
        for j in range(A.shape[0]):
            if bool(is_cone[j]):
                continue
            support = V @ A[j]
            active = np.where(np.abs(support - b[j]) <= tol)[0]
            if active.size < 2:
                continue
            t = np.array([-A[j, 1], A[j, 0]], dtype=float)
            nrm = float(np.linalg.norm(t))
            if nrm <= 1e-12:
                continue
            t = t / nrm
            proj = V[active] @ t
            p0 = V[active[np.argmin(proj)]]
            p1 = V[active[np.argmax(proj)]]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=edge_color, linewidth=1.3, zorder=5)

        # Unbounded outer directions induced by cone facet normals.
        step = 1e-3 * max(1.0, float(np.linalg.norm(span)))
        cone_normals = cls._compute_cone_facet_normals_2d(G)

        def _in_recession_cone(d: np.ndarray) -> bool:
            if cone_normals.size == 0:
                return True
            return bool(np.all(cone_normals @ d <= 50.0 * tol))

        for j in range(A.shape[0]):
            if not bool(is_cone[j]):
                continue
            support = V @ A[j]
            active = np.where(np.abs(support - b[j]) <= tol)[0]
            if active.size == 0:
                continue

            anchor = V[active[np.argmax(support[active])]]
            tangents = [
                np.array([-A[j, 1], A[j, 0]], dtype=float),
                np.array([A[j, 1], -A[j, 0]], dtype=float),
            ]
            ray_dir: Optional[np.ndarray] = None
            for t in tangents:
                nrm = float(np.linalg.norm(t))
                if nrm <= 1e-12:
                    continue
                t_unit = t / nrm
                if not _in_recession_cone(t_unit):
                    continue
                test_pt = anchor + step * t_unit
                end_pt = _ray_endpoint_on_viewport(anchor, t_unit)
                if end_pt is None:
                    continue
                if bool(np.all(A @ test_pt <= b + 20.0 * tol)) and bool(np.all(A @ end_pt <= b + 100.0 * tol)):
                    ray_dir = t_unit
                    break
            if ray_dir is None:
                continue

            end = _ray_endpoint_on_viewport(anchor, ray_dir)
            if end is None:
                continue
            ax.plot([anchor[0], end[0]], [anchor[1], end[1]], color=edge_color, linewidth=1.3, zorder=5)

    @classmethod
    def _draw_unbounded_set_fill_2d(
        cls,
        ax: Axes,
        vertices: np.ndarray,
        generators: np.ndarray,
        *,
        fill_color: str,
        fill_alpha: float,
        edge_color: str,
        label: Optional[str] = None,
        draw_frontier: bool = True,
        grid_size: int = 260,
        viewport: Optional[tuple[float, float, float, float]] = None,
    ) -> None:
        """Draw conv(vertices)+cone(generators) as a filled region clipped to view."""
        V = np.asarray(vertices, dtype=float)
        if V.ndim != 2 or V.shape[1] != 2 or len(V) == 0:
            return

        G = np.asarray(generators, dtype=float)
        if G.size == 0:
            G = np.empty((0, 2), dtype=float)

        if viewport is None:
            x_min, x_max, y_min, y_max = cls._compute_viewport_2d(V, G)
        else:
            x_min, x_max, y_min, y_max = viewport

        A, b, is_cone = cls._build_unbounded_set_inequalities_2d(V, G)

        xs = np.linspace(x_min, x_max, grid_size)
        ys = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(xs, ys)

        mask = np.ones_like(X, dtype=bool)
        for j in range(A.shape[0]):
            mask &= (A[j, 0] * X + A[j, 1] * Y <= b[j] + 1e-7)

        Z = mask.astype(float)
        ax.contourf(
            X,
            Y,
            Z,
            levels=[0.5, 1.5],
            colors=[fill_color],
            alpha=fill_alpha,
            antialiased=True,
        )

        if label is not None:
            ax.scatter([], [], color=fill_color, alpha=fill_alpha, label=label)

        if draw_frontier:
            cls._draw_unbounded_boundary_2d(
                ax,
                vertices=V,
                generators=G,
                A=A,
                b=b,
                is_cone=is_cone,
                edge_color=edge_color,
                viewport=viewport,
            )

    def _img_F_direct_constraints(self, y: cp.Variable) -> List[cp.Constraint]:
        """Direct CVX description of img F using only instance variables."""
        n = self.graph.n
        # aux = [x_ref, z1, z2]
        aux = cp.Variable(3 * n, name="imgF_aux")
        x_ref = aux[0:n]
        z1 = aux[n:2 * n]
        z2 = aux[2 * n:3 * n]

        # img F = { y : ∃ x_ref,z1,z2 satisfying graph constraints }
        constraints = [
            y[0] <= self.p1 * (self.r1 @ z1) + self.p2 * (self.r2 @ z2),
            y[1] <= -self.p1 * cp.quad_form(z1, self.Q1) - self.p2 * cp.quad_form(z2, self.Q2),
            z1 >= 0,
            z2 >= 0,
            cp.sum(z1) == 1,
            cp.sum(z2) == 1,
            cp.norm(x_ref - z1, "inf") <= self.tau,
            cp.norm(x_ref - z2, "inf") <= self.tau,
            x_ref >= 0,
            x_ref <= 1
        ]
        return cast(List[cp.Constraint], constraints)

    def _build_imgF_set(self) -> ConvexSet:
        """Build img F as a convex set in R^2 (represented as x-variable of ConvexSet)."""
        n = self.graph.n
        # shadow variable layout: [x_ref, z1, z2]
        set_s = ConvexSet(
            n=2,
            q=0,
            m=3 * n,
            name="ImgFSet",
            recession_cone_generators=[[-1.0, 0.0], [0.0, -1.0]],
        )
        set_s.add_constraint_fn(lambda x, y, z: [
            x[0] <= self.p1 * (self.r1 @ z[n:2 * n]) + self.p2 * (self.r2 @ z[2 * n:3 * n]),
            x[1] <= -self.p1 * cp.quad_form(z[n:2 * n], self.Q1) - self.p2 * cp.quad_form(z[2 * n:3 * n], self.Q2),
            z[n:2 * n] >= 0,
            z[2 * n:3 * n] >= 0,
            cp.sum(z[n:2 * n]) == 1,
            cp.sum(z[2 * n:3 * n]) == 1,
            cp.norm(z[0:n] - z[n:2 * n], "inf") <= self.tau,
            cp.norm(z[0:n] - z[2 * n:3 * n], "inf") <= self.tau,
            z[0:n] >= 0,
            z[0:n] <= 1
        ])
        return set_s

    def _build_Fx_set(self, x_val: np.ndarray) -> ConvexSet:
        """Build F(x_val) as a convex set in R^2 (represented as x-variable of ConvexSet)."""
        n = self.graph.n
        x_val = np.asarray(x_val, dtype=float).ravel()
        set_s = ConvexSet(
            n=2,
            q=0,
            m=2 * n,
            name="FxSet",
            recession_cone_generators=[[-1.0, 0.0], [0.0, -1.0]],
        )
        set_s.add_constraint_fn(lambda x, y, z: [
            x[0] <= self.p1 * (self.r1 @ z[0:n]) + self.p2 * (self.r2 @ z[n:2 * n]),
            x[1] <= -self.p1 * cp.quad_form(z[0:n], self.Q1) - self.p2 * cp.quad_form(z[n:2 * n], self.Q2),
            z >= 0,
            cp.sum(z[0:n]) == 1,
            cp.sum(z[n:2 * n]) == 1,
            cp.norm(x_val - z[0:n], "inf") <= self.tau,
            cp.norm(x_val - z[n:2 * n], "inf") <= self.tau
        ])
        return set_s

    def _find_feasible_point_in_set(self, set_s: ConvexSet) -> np.ndarray:
        """Find one feasible point of a ConvexSet for initializing approximation."""
        x = cp.Variable(set_s.n, name="feas_x")
        y = cp.Variable(set_s.q, name="feas_y") if set_s.q > 0 else None
        z = cp.Variable(set_s.m, name="feas_z") if set_s.m > 0 else None
        constraints = set_s.make_constraints(x, y, z)
        problem = cp.Problem(cp.Maximize(0), constraints)

        self._solve_problem(problem)
        if x.value is not None:
            return np.asarray(x.value, dtype=float).ravel()

        raise RuntimeError("Could not compute a feasible point for set approximation.")

    def _approximate_set_vertices(
        self,
        set_s: ConvexSet,
        *,
        eps: Optional[float] = None,
        y0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Approximate a ConvexSet with the γε-optimizer algorithm from CSOP and return its approximation vertices."""
        if y0 is None:
            y0 = self._find_feasible_point_in_set(set_s)

        eps_eff = 0.001 if eps is None else float(eps)
        if eps_eff <= 0:
            raise ValueError(f"eps must be > 0, got {eps_eff}.")

        approx = ConvexSetApproximator(set_s, solver=self.solver)
        result = approx.approximate(y=np.asarray(y0, dtype=float), eps=eps_eff)
        verts = result["approx"].vertices
        if verts is None or len(verts) == 0:
            raise RuntimeError("Set approximation produced no vertices.")
        return np.asarray(verts, dtype=float)

    def plot(
        self,
        approximation: Optional[Approximation] = None,
        y_choice: Optional[np.ndarray] = None,
        *,
        eps: Optional[float] = None,
        ax: Optional[Axes] = None,
        img_color: str = "steelblue",
        img_alpha: float = 0.30,
        img_edge_color: str = "steelblue",
        img_label: str = r"img $F$",
        approx_color: str = "tomato",
        approx_alpha: float = 0.25,
        approx_edge_color: str = "tomato",
        approx_vertex_color: str = "tomato",
        approx_vertex_size: float = 10,
    ) -> Axes:
        """Plot  img F = ⋃_x F(x)  and approximation as overlay

        Parameters
        approximation : Approximation, optional
            Approximation object from run().  Its convex hull is drawn on
            top of  img F.
        img_F_fn : callable (y_var: cp.Variable) -> List[cp.Constraint], optional
            Direct CVX description of  img F.
        ax : matplotlib.axes.Axes, optional
            Target axes; a new figure is created when omitted.
        epsilon : float
            tolerance ε for gammaEps-optimizer algorithm; proportional to precision of approximation.
        """
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        # img F (approximated via ConvexSetApproximator)
        if y_choice is not None:
            y0_img = np.asarray(y_choice, dtype=float).ravel()
        else:
            y0_img = None
        eps_eff = 0.001
        pts_arr = self._approximate_set_vertices(self._build_imgF_set(), y0=y0_img, eps=eps_eff)
        recession_generators = np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=float)
        pts_arr = self._prune_vertices_for_plot(pts_arr, recession_generators)
        plot_viewport = self._compute_viewport_2d(pts_arr, recession_generators)
        viewport_vertices: List[np.ndarray] = [pts_arr]

        if approximation is not None and approximation.vertices is not None:
            V_view = self._prune_vertices_for_plot(
                np.asarray(approximation.vertices, dtype=float),
                np.asarray(
                    getattr(approximation, "recession_cone_generators", np.empty((0, 2))),
                    dtype=float,
                ),
            )
            if V_view.ndim == 2 and V_view.shape[1] == 2 and len(V_view) > 0:
                viewport_vertices.append(V_view)

        plot_viewport = self._expand_viewport_with_vertices(plot_viewport, viewport_vertices)
        plot_viewport = self._square_viewport(plot_viewport)
        self._plot_viewport = plot_viewport
        self._draw_unbounded_set_fill_2d(
            ax,
            vertices=pts_arr,
            generators=recession_generators,
            fill_color=img_color,
            fill_alpha=img_alpha,
            edge_color=img_edge_color,
            label=img_label,
            viewport=plot_viewport,
        )

        # approximation
        if approximation is not None:
            V_raw = approximation.vertices
            if V_raw is None:
                V = None
            else:
                V = self._prune_vertices_for_plot(
                    np.asarray(V_raw, dtype=float),
                    np.asarray(
                        getattr(approximation, "recession_cone_generators", np.empty((0, 2))),
                        dtype=float,
                    ),
                )
            x_opt = approximation.feasiblePoint
            x_str = np.array2string(x_opt, precision=3, separator=", ", suppress_small=True)
            if y_choice is not None:
                y_str = np.array2string(y_choice, precision=3, separator=", ", suppress_small=True)
                approx_label = rf"Approximation $I$ of $F(x)$ with $\gamma \varepsilon$-optimizer x = {x_str} and starting point $y_0={y_str} \in F(x)$"
            else:
                approx_label = rf"Approximation $I$ of $F(x)$ with $\gamma \varepsilon$-optimizer x = {x_str}$"
            if V is not None and len(V) >= 3:
                self._draw_unbounded_set_fill_2d(
                    ax,
                    vertices=V,
                    generators=np.asarray(
                        getattr(approximation, "recession_cone_generators", np.empty((0, 2))),
                        dtype=float,
                    ),
                    fill_color=approx_color,
                    fill_alpha=approx_alpha,
                    edge_color=approx_edge_color,
                    label=approx_label,
                    viewport=plot_viewport,
                )
            elif V is not None and len(V) == 2:
                self._draw_unbounded_set_fill_2d(
                    ax,
                    vertices=V,
                    generators=np.asarray(
                        getattr(approximation, "recession_cone_generators", np.empty((0, 2))),
                        dtype=float,
                    ),
                    fill_color=approx_color,
                    fill_alpha=approx_alpha,
                    edge_color=approx_edge_color,
                    label=approx_label,
                    viewport=plot_viewport,
                )
            elif V is not None:
                ax.scatter(V[:, 0], V[:, 1],
                           color=approx_vertex_color, s=approx_vertex_size,
                           zorder=5, label=approx_label)
            if V is not None:
                ax.scatter(V[:, 0], V[:, 1],
                           color=approx_vertex_color, s=approx_vertex_size, zorder=5)

        ax.set_xlabel(r"$y_1$ (yield)")
        ax.set_ylabel(r"$y_2$ ($-$risk)")
        ax.set_xlim(plot_viewport[0], plot_viewport[1])
        ax.set_ylim(plot_viewport[2], plot_viewport[3])
        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        return ax

    def plot_Fx(
        self,
        x_list: List[np.ndarray],
        *,
        eps: Optional[float] = None,
        img_F_fn: Optional[Callable[[cp.Variable], List[cp.Constraint]]] = None,
        ax: Optional[Axes] = None,
        img_color: str = "steelblue",
        img_alpha: float = 0.30,
        img_edge_color: str = "steelblue",
        img_label: str = r"img $F$",
        fx_colors: Optional[List[str]] = None,
        fx_alpha: float = 0.30,
        csv_dir: Optional[str] = None,
        save_to_csv: bool = False,
    ) -> Axes:
        """Plot F(x) for one or two fixed portfolios x, overlaid on img F.

        For each x in x_list, checks sum(x) == 1 and x >= 0, then sweeps
        support directions to trace the boundary of F(x) and fills the
        resulting convex hull on top of img F.

        Parameters
        ----------
        x_list : list of np.ndarray, shape (n,)
            One or two fixed reference portfolios.
        img_F_fn : callable (y_var: cp.Variable) -> list, optional
            Direct CVX description of img F.  When given, img F is drawn as
            background before the F(x) sets.
        ax : matplotlib.axes.Axes, optional
            Target axes; a new figure is created when omitted.
        img_color, img_alpha, img_edge_color, img_label
            Styling for img F (only used when img_F_fn is given).
        fx_colors : list of str, optional
            One color per entry in x_list.  Defaults to
            ["tomato", "forestgreen", "darkorange", "purple"].
        fx_alpha : float
            Fill transparency for all F(x) sets.
        epsilon : float
            tolerance ε for gammaEps-optimizer algorithm; proportional to precision of approximation.
        """
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None

        _default_colors = ["tomato", "forestgreen", "darkorange", "purple"]
        if fx_colors is None:
            fx_colors = _default_colors[: len(x_list)]

        n = self.graph.n

        # validate x vectors
        for k, x_val in enumerate(x_list):
            x_val = np.asarray(x_val, dtype=float).ravel()
            if x_val.shape != (n,):
                raise ValueError(f"x_list[{k}] must have shape ({n},), got {x_val.shape}.")
            if np.any(x_val < -1e-8):
                raise ValueError(
                    f"x_list[{k}] must be >= 0, but has negative entries: {x_val[x_val < 0]}."
                )
            if abs(x_val.sum() - 1.0) > 1e-4:
                raise ValueError(
                    f"x_list[{k}] must satisfy sum(x) = 1, but sum = {x_val.sum():.6f}."
                )

        pts_imgF: List[np.ndarray] = []
        all_pts_Fx: List = []
        eps_eff = 0.001

        # Compute all vertex sets first so viewport can include every plotted vertex.
        img_vertices = self._approximate_set_vertices(self._build_imgF_set(), eps=eps_eff)
        recession_generators = np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=float)
        img_vertices = self._prune_vertices_for_plot(img_vertices, recession_generators)
        pts_imgF = [v.copy() for v in img_vertices]
        fx_sets: List[tuple[np.ndarray, str, str, np.ndarray]] = []

        for k, x_val in enumerate(x_list):
            x_val = np.asarray(x_val, dtype=float).ravel()
            color = fx_colors[k] if k < len(fx_colors) else _default_colors[k % len(_default_colors)]
            x_str = np.array2string(x_val, precision=3, separator=", ", suppress_small=True)

            fx_vertices = self._approximate_set_vertices(self._build_Fx_set(x_val), eps=eps_eff)
            fx_vertices = self._prune_vertices_for_plot(fx_vertices, recession_generators)
            pts_Fx: List[np.ndarray] = [v.copy() for v in fx_vertices]
            if not pts_Fx:
                print(f"Warning: No feasible points found for x_list[{k}] = {x_str}. Skipping.")
                continue

            all_pts_Fx.append((x_val.copy(), pts_Fx))
            fx_sets.append((x_val.copy(), color, x_str, np.asarray(pts_Fx, dtype=float)))

        plot_viewport = self._compute_viewport_2d(img_vertices, recession_generators)
        viewport_vertices = [img_vertices] + [pts for _, _, _, pts in fx_sets]
        plot_viewport = self._expand_viewport_with_vertices(plot_viewport, viewport_vertices)
        plot_viewport = self._square_viewport(plot_viewport)
        self._plot_viewport = plot_viewport

        # draw img F background
        if len(pts_imgF) >= 2:
            pts_arr = np.array(pts_imgF)
            self._draw_unbounded_set_fill_2d(
                ax,
                vertices=pts_arr,
                generators=recession_generators,
                fill_color=img_color,
                fill_alpha=img_alpha,
                edge_color=img_edge_color,
                label=img_label,
                viewport=plot_viewport,
            )

        # draw F(x) for each fixed x
        for k, (x_val, color, x_str, pts_arr) in enumerate(fx_sets):
            label = rf"$F(x_{k+1})$, $x_{k+1} = {x_str}$"
            if len(pts_arr) >= 2:
                self._draw_unbounded_set_fill_2d(
                    ax,
                    vertices=pts_arr,
                    generators=recession_generators,
                    fill_color=color,
                    fill_alpha=fx_alpha,
                    edge_color=color,
                    label=label,
                    viewport=plot_viewport,
                )
            else:
                ax.scatter(pts_arr[:, 0], pts_arr[:, 1], color=color, s=20, label=label)

        ax.set_xlabel(r"$y_1$ (yield)")
        ax.set_ylabel(r"$y_2$ ($-$risk)")
        ax.set_xlim(plot_viewport[0], plot_viewport[1])
        ax.set_ylim(plot_viewport[2], plot_viewport[3])
        ax.legend()
        ax.set_aspect("equal", adjustable="box")

        # --- CSV export ---
        if save_to_csv and csv_dir is not None:
            os.makedirs(csv_dir, exist_ok=True)
            vertex_header = ["vertex_idx", "y1", "y2"]

            # img F file - only created if file does not yet exist
            if pts_imgF:
                imgF_path = os.path.join(csv_dir, "imgF.csv")
                if not os.path.exists(imgF_path):
                    pts_arr_imgF = np.array(pts_imgF)
                    if len(pts_arr_imgF) >= 3:
                        hull_g = ConvexHull(pts_arr_imgF)
                        verts_grF = pts_arr_imgF[hull_g.vertices]
                    else:
                        verts_grF = pts_arr_imgF
                    verts_grF = self._sort_vertices_clockwise(verts_grF)
                    with open(imgF_path, "w", newline="", encoding="utf-8") as fh:
                        writer = csv.writer(fh, delimiter=";")
                        writer.writerow(vertex_header)
                        for vi, v in enumerate(verts_grF):
                            writer.writerow([vi, v[0], v[1]])
                    print(f"Saved img F vertices to '{imgF_path}'.")

            # one CSV per F(x)
            for x_val_k, pts_k in all_pts_Fx:
                x_str_fname = ",".join(f"{v:.4g}" for v in x_val_k)
                fname = f"F({x_str_fname})_approx.csv"
                fpath = os.path.join(csv_dir, fname)
                if not os.path.exists(fpath):
                    pts_arr_k = np.array(pts_k)
                    if len(pts_arr_k) >= 3:
                        hull_k = ConvexHull(pts_arr_k)
                        verts_k = pts_arr_k[hull_k.vertices]
                    else:
                        verts_k = pts_arr_k
                    verts_k = self._sort_vertices_clockwise(verts_k)
                    with open(fpath, "w", newline="", encoding="utf-8") as fh:
                        writer = csv.writer(fh, delimiter=";")
                        writer.writerow(vertex_header)
                        for vi, v in enumerate(verts_k):
                            writer.writerow([vi, v[0], v[1]])
                    print(f"Saved F(x) vertices to '{fpath}'.")
                else:
                    print(f"F(x) file already exists, skipping: '{fpath}'.")

        return ax

    def plot_iterations(
        self,
        approximations: List[Approximation],
        y_choice: Optional[np.ndarray] = None,
        *,
        gamma_upper_bound: Optional[float] = None,
        eps: Optional[float] = None,
        ncols: int = 3,
        figsize_per_ax: tuple = (3.5, 3.5),
        suptitle: Optional[str] = None,
        img_color: str = "steelblue",
        img_alpha: float = 0.30,
        img_edge_color: str = "steelblue",
        approx_color: str = "tomato",
        approx_alpha: float = 0.25,
        approx_edge_color: str = "tomato",
        approx_vertex_color: str = "tomato",
        approx_vertex_size: float = 10,
        vertex_merge_tol: float = 1e-8,
        csv_dir: Optional[str] = None,
        save_to_csv: bool = False,
    ) -> Figure:
        """Plot the evolution of the approximation across algorithm iterations.

        Parameters
        ----------
        approximations : List[Approximation]
            Snapshots from the on_iteration callback of run().
            Index 0 is the initial approximation (before the loop),
            subsequent entries correspond to algorithm steps 1, 2, ...
        y_choice : np.ndarray, optional
            Starting point y₀ used in run().
        ncols : int
            Number of columns in the subplot grid.
        figsize_per_ax : tuple
            (width, height) in inches per subplot cell.
        suptitle : str, optional
            Figure-level title.  When *None* (default) an informative title
            is generated automatically from ``self.epsilon``, *y_choice*, and
            the gammaEps-optimizer x* of the last approximation snapshot.
        csv_dir : str, optional
            If given, exports one CSV file per iteration into this directory.
            The directory is created if it does not exist.  Each filename
            includes the iteration index, label, and a run timestamp.
        save_to_csv : bool
            Set to True to enable CSV export.  Requires csv_dir to be set.
            Default is False.
        epsilon : float
            tolerance ε for gammaEps-optimizer algorithm; proportional to precision of approximation.
        vertex_merge_tol : float
            Tolerance for merging numerically near-identical vertices in snapshots.
            This keeps the displayed marker count and the title count aligned.
        """
        # approximate img F once and reuse across all subplots
        y0_img = np.asarray(y_choice, dtype=float).ravel() if y_choice is not None else None
        eps_eff = self.epsilon if eps is None else float(eps)
        pts_arr = self._approximate_set_vertices(self._build_imgF_set(), y0=y0_img, eps=eps_eff)

        # Prepare plotted approximation vertices first, then choose one viewport for all subplots.
        prepared_vertices: List[Optional[np.ndarray]] = []
        for apx in approximations:
            V_raw = apx.vertices
            if V_raw is None:
                prepared_vertices.append(None)
                continue
            V = self._deduplicate_vertices_for_plot(
                np.asarray(V_raw, dtype=float),
                tol=vertex_merge_tol,
            )
            V = self._prune_vertices_for_plot(
                V,
                np.asarray(
                    getattr(apx, "recession_cone_generators", np.empty((0, 2))),
                    dtype=float,
                ),
                tol=vertex_merge_tol,
            )
            prepared_vertices.append(V if len(V) > 0 else None)

        recession_generators = np.asarray([[-1.0, 0.0], [0.0, -1.0]], dtype=float)
        plot_viewport = self._compute_viewport_2d(pts_arr, recession_generators)
        viewport_vertices = [pts_arr] + [V for V in prepared_vertices if V is not None]
        plot_viewport = self._expand_viewport_with_vertices(plot_viewport, viewport_vertices)
        plot_viewport = self._square_viewport(plot_viewport)
        self._plot_viewport = plot_viewport
        # draw img F per-axis via unbounded fill helper (conv(vertices) + recession cone)

        # subplot grid
        n = len(approximations)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * figsize_per_ax[0], nrows * figsize_per_ax[1]),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        for i, apx in enumerate(approximations):
            ax = axes_flat[i]
            first = (i == 0)

            # img F background
            self._draw_unbounded_set_fill_2d(
            ax,
            vertices=pts_arr,
            generators=recession_generators,
            fill_color=img_color,
            fill_alpha=img_alpha,
            edge_color=img_edge_color,
            label=r"img $F$" if first else None,
            viewport=plot_viewport,
            )

            # approximation
            V = prepared_vertices[i]

            if V is not None and len(V) >= 3:
                self._draw_unbounded_set_fill_2d(
                    ax,
                    vertices=V,
                    generators=np.asarray(
                        getattr(apx, "recession_cone_generators", np.empty((0, 2))),
                        dtype=float,
                    ),
                    fill_color=approx_color,
                    fill_alpha=approx_alpha,
                    edge_color=approx_edge_color,
                    label=r"Approx. $I$" if first else None,
                    viewport=plot_viewport,
                )
                ax.scatter(V[:, 0], V[:, 1],
                           color=approx_vertex_color, s=approx_vertex_size, zorder=5)
            elif V is not None and len(V) == 2:
                self._draw_unbounded_set_fill_2d(
                    ax,
                    vertices=V,
                    generators=np.asarray(
                        getattr(apx, "recession_cone_generators", np.empty((0, 2))),
                        dtype=float,
                    ),
                    fill_color=approx_color,
                    fill_alpha=approx_alpha,
                    edge_color=approx_edge_color,
                    label=r"Approx. $I$" if first else None,
                    viewport=plot_viewport,
                )
                ax.scatter(V[:, 0], V[:, 1],
                           color=approx_vertex_color, s=approx_vertex_size, zorder=5)
            elif V is not None and len(V) == 1:
                ax.scatter(V[:, 0], V[:, 1],
                           color=approx_vertex_color, s=approx_vertex_size,
                           zorder=5, label=r"Approx. $I$" if first else None)

            if V is None:
                n_verts = 0
            else:
                n_verts = int(V.shape[0])
            title = ("Initial" if i == 0 else f"Update {i}") + f"  ({n_verts} vert.)"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(r"$y_1$", fontsize=8)
            ax.set_ylabel(r"$y_2$", fontsize=8)
            ax.set_xlim(plot_viewport[0], plot_viewport[1])
            ax.set_ylim(plot_viewport[2], plot_viewport[3])
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelsize=7)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        handles, labels_ = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, 0.005))

        # build auto suptitle with run info
        if suptitle is None:
            title_lines = [rf"Approximation evolution  ($\varepsilon = {self.epsilon}$)"]
            info_parts: List[str] = []
            if gamma_upper_bound is not None:
                if np.isinf(gamma_upper_bound):
                    info_parts.append(r"upper bound $\gamma \leq \infty$")
                else:
                    info_parts.append(rf"upper bound $\gamma \leq = {float(gamma_upper_bound):.4g}$")
            if y_choice is not None:
                y_str = np.array2string(y_choice, precision=3, separator=", ", suppress_small=True)
                info_parts.append(rf"starting point $y_0 = {y_str}$")
            if approximations:
                x_last = approximations[-1].feasiblePoint
                if x_last is not None:
                    x_str = np.array2string(x_last, precision=3, separator=", ", suppress_small=True)
                    info_parts.append(rf"$\gamma \varepsilon$-optimizer $x^* = {x_str}$")
            if info_parts:
                title_lines.append(",  ".join(info_parts))
            suptitle = "\n".join(title_lines)

        fig.suptitle(suptitle, fontsize=10, y=0.99)
        fig.tight_layout(rect=(0, 0.08, 1, 0.90))

        # --- CSV export (one file per iteration + one problem info file) ---
        if save_to_csv and csv_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(csv_dir, timestamp)
            os.makedirs(run_dir, exist_ok=True)
            n_assets = self.graph.n

            # -- per-iteration CSVs: only vertices (y1, y2) and current x --
            x_cols = [f"x_{j}" for j in range(n_assets)]
            iter_header = ["iteration", "iteration_label", "vertex_idx", "vertex y[0]", "vertex y[1]", "n_vertices"] + x_cols

            for i, apx in enumerate(approximations):
                label = "initial" if i == 0 else f"update_{i:03d}"
                fname = os.path.join(run_dir, f"iter_{i:03d}_{label}.csv")
                V = prepared_vertices[i]
                x_fp = apx.feasiblePoint
                n_v = len(V) if V is not None else 0
                V_sorted = self._sort_vertices_clockwise(V) if V is not None else None
                x_vals = list(x_fp) if x_fp is not None else [""] * n_assets
                with open(fname, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh, delimiter=";")
                    writer.writerow(iter_header)
                    if V_sorted is not None:
                        for vi, vert in enumerate(V_sorted):
                            writer.writerow([i, label, vi, vert[0], vert[1], n_v] + x_vals)
                    else:
                        writer.writerow([i, label, "", "", "", 0] + x_vals)

            # -- problem info CSV: fully describes the problem setting --
            info_fname = os.path.join(run_dir, "problem_info.csv")
            info_rows: list = [
                ["parameter", "value"],
                ["timestamp", timestamp],
                ["n_assets", n_assets],
                ["epsilon", self.epsilon],
                ["tau", self.tau],
                ["p1", self.p1],
                ["p2", self.p2],
                ["solver", self.solver if self.solver is not None else "auto"],
            ]
            # r1 vector
            for j, rj in enumerate(self.r1.flatten()):
                info_rows.append([f"r1_{j}", rj])
            # r2 vector
            for j, rj in enumerate(self.r2.flatten()):
                info_rows.append([f"r2_{j}", rj])
            # Q1 matrix
            for ii in range(self.Q1.shape[0]):
                for jj in range(self.Q1.shape[1]):
                    info_rows.append([f"Q1_{ii}_{jj}", self.Q1[ii, jj]])
            # Q2 matrix
            for ii in range(self.Q2.shape[0]):
                for jj in range(self.Q2.shape[1]):
                    info_rows.append([f"Q2_{ii}_{jj}", self.Q2[ii, jj]])
            # starting point y0
            if y_choice is not None:
                for j, yj in enumerate(y_choice):
                    info_rows.append([f"target point y0_{j}", yj])

            with open(info_fname, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh, delimiter=";")
                writer.writerows(info_rows)

            print(f"Iteration CSVs saved to '{run_dir}/' ({len(approximations)} iteration files + problem_info.csv).")

        return fig

    def __repr__(self) -> str:
        return f"PortfolioOpt(n={self.graph.n}, tau={self.tau}, epsilon={self.epsilon})"

############################################################################################################################################################

# Example with data

if __name__ == "__main__":

    tau = 0.2

    # Scenario 1: Boom  (Asset 1 = bond, Asset 2 = stock)
    p1 = 0.5
    r1 = np.array([0.8, 1.5])
    Q1 = np.array([
        [1.0, 0.6],
        [0.6, 2.2]
    ])

    # Scenario 2: Recession
    p2 = 0.5
    r2 = np.array([0.8, 0.6])
    Q2 = np.array([
        [1.0, 0.6],
        [0.6, 2.3]
    ])

    # define initial point to be in F(x) with gammaEps-optimizer x to be returned 
    # suggestions: for equal-weight portfolio z0=[0.5, 0.5] in each scenario: y0 = [0.925, -1.1125]
    # or obtain e.g. another other gammaEps-optimizer for y0 = [0.87295635, -0.92557209] (boundary point of img F)
    # or: [1.05, -1.4]
    # or set y0 = None to auto-select a feasible point in img F
    y0 = np.array([0.87295635, -0.92557209])

    epsilon = 0.01
    opt = PortfolioOptScen(
        r1=r1,
        Q1=Q1,
        p1=p1,
        r2=r2,
        Q2=Q2,
        p2=p2,
        tau=tau,
        epsilon=epsilon,
        solver="SCS",
    )
    snapshots: List[Approximation] = []
    computeGammaUpperBound = True
    result = opt.run(
        y0,
        verbose=True,
        compute_gamma_upper_bound=computeGammaUpperBound,
        on_iteration=lambda i, apx: snapshots.append(apx),
    )
    gamma_upper = result.get("gamma_upper_bound", result.get("L_upper_bound"))
    if gamma_upper is None:
        gamma_title = "L=n/a"
    elif np.isinf(gamma_upper):
        gamma_title = "L=inf"
    else:
        gamma_title = rf"$\gamma <=$ L={float(gamma_upper):.4g}"

    ##########################################################################################################################
    
    # Plotting of the results 
    saveIterToCsv = False

    ax = opt.plot(result["approx"], result["y_choice"])
    ax.set_title(rf"img $F$ with $\gamma \varepsilon$-approximation of F(x) ($\varepsilon$ = {epsilon}, {gamma_title})")
    plt.tight_layout()

    fig = opt.plot_iterations(snapshots, result["y_choice"],
                              gamma_upper_bound=gamma_upper,
                              csv_dir="portfolio_algorithm_csv", save_to_csv=saveIterToCsv)
    plt.show()

    ##########################################################################################################################

    # Optional: plotting of some values F(x) to visualize flexibility effect
    plotFx = False
    saveFxToCsv = False

    x1 = np.array([0.621, 0.379]) 
    x2 = np.array([0.85, 0.15])   

    if(plotFx):
        #e.g. choose 0.01-optimizers x1 = [0.621, 0.379] and x2 = [0.358, 0.642] to visualize non-comparable sets
        # or: dominance and visibly more flexibility for x1 = [0.85, 0.15] compared to x2 = [0.8, 0.2]
        ax2 = opt.plot_Fx(
            [x1, x2],
            eps = 0.0001,
            csv_dir="portfolio_plots_csv",
            save_to_csv=saveFxToCsv,
        )
        ax2.set_title(rf"img $F$ with $F(x)$ for selected portfolios ($\tau = {tau}$)")
        plt.tight_layout()
        plt.show()
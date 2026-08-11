from __future__ import annotations
from typing import Optional, cast
import numpy as np
import cvxpy as cp
from cvxpy.constraints.constraint import Constraint

from scipy.spatial import ConvexHull
from epsopt import Graph


class Approximation:
    """Polyhedral approximation of the image of a set-valued function F at some point x.

    Stores an (optionally unbounded) polyhedral approximation in the form
    ``P + recc`` where
    ``P = conv(vertices)`` and ``recc`` is the recession cone stored in
    the associated graph.

    - ``vertices`` (or ``vertP``): generators of the bounded part ``P``
    - ``normals``: unit outer normals of facets of ``P + recc``

    graph : Graph
        The graph object this approximation belongs to.
    """

    # instance attributes
    _x: np.ndarray                        # feasible point x ∈ dom F
    _vertices: Optional[np.ndarray]       # V-representation: rows are vertices in R^q
    _normals: Optional[np.ndarray]        # outer normals: rows are unit facet normals in R^q
    _cone_facet_normals: np.ndarray       # cached facet normals of recession cone in R^q
    _graph: Graph

    def __init__(self, x: np.ndarray, graph: Graph, epsilon: float = np.inf) -> None:
        self._x = x
        self._epsilon = float(epsilon)
        self._vertices = None
        self._graph = graph
        self._cone_facet_normals = self._compute_cone_facet_normals_once()
        # Keep cone-facet normals in normals from the beginning on.
        self._normals = self._cone_facet_normals.copy()

    @property
    def recession_cone_generators(self) -> np.ndarray:
        """Recession-cone generators as array of shape (r, q)."""
        gens = getattr(self._graph, "recession_cone_generators", [])
        if not gens:
            return np.empty((0, self._graph.q), dtype=float)
        return np.asarray(gens, dtype=float)

    @property
    def cone_facet_normals(self) -> np.ndarray:
        """Cached facet normals of the recession cone."""
        return self._cone_facet_normals
    
    @property
    def feasiblePoint(self) -> np.ndarray:
        return self._x


    @property
    def vertices(self) -> Optional[np.ndarray]:
        """V-representation: array of shape (k, q), or None."""
        return self._vertices

    @property
    def vertP(self) -> Optional[np.ndarray]:
        """Alias for vertices of the bounded part P in P + recc."""
        return self._vertices

    @property
    def normals(self) -> Optional[np.ndarray]:
        """Minimal system of outer normals: array of shape (m, q), or None.

        Each row is a unit outer normal of a facet of the approximation polytope.
        """
        return self._normals


    def set_vertices(self, vertices: np.ndarray) -> None:
        """Set the V-representation by inputting an array
        vertices : np.ndarray

        Normals are updated automatically afterwards calling self._v_to_normals().
        """
        self._vertices = np.asarray(vertices, dtype=float)
        self._prune_vertices_to_extreme_points()
        self._v_to_normals()

    def add_vertex(self, y: np.ndarray) -> None:
        """Extend the V-representation by one new point
        y : np.ndarray, shape (q,)

        Normals are updated automatically afterwards calling self._v_to_normals().
        """
        y = np.asarray(y, dtype=float).ravel()
        if self._vertices is None:
            self._vertices = y[np.newaxis, :]
        else:
            self._vertices = np.vstack([self._vertices, y])
        self._prune_vertices_to_extreme_points()
        # normals have changed
        self._normals = None
        self._v_to_normals()

    def _prune_vertices_to_extreme_points(self, tol: float = 1e-10) -> None:
        """Keep only extreme points of ``conv(vertices) + recc``.

        A point is removed if it can be represented by a convex combination
        of the other points plus a recession-cone direction.
        """
        if self._vertices is None or len(self._vertices) == 0:
            return

        V = np.asarray(self._vertices, dtype=float)
        q = V.shape[1]
        cone = self.recession_cone_generators

        # Remove exact/near duplicates first while preserving original rows.
        rounded = np.round(V, decimals=12)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        V = V[np.sort(idx)]
        if len(V) <= 1:
            self._vertices = V
            return

        if q == 1:
            vmin = float(np.min(V[:, 0]))
            vmax = float(np.max(V[:, 0]))
            if cone.size == 0:
                if abs(vmax - vmin) <= tol:
                    self._vertices = np.array([[vmin]], dtype=float)
                else:
                    self._vertices = np.array([[vmin], [vmax]], dtype=float)
                return

            rays = cone[:, 0]
            has_pos = bool(np.any(rays > tol))
            has_neg = bool(np.any(rays < -tol))
            if has_pos and has_neg:
                self._vertices = np.array([[vmin]], dtype=float)
                return
            if has_pos:
                self._vertices = np.array([[vmin]], dtype=float)
                return
            if has_neg:
                self._vertices = np.array([[vmax]], dtype=float)
                return
            if abs(vmax - vmin) <= tol:
                self._vertices = np.array([[vmin]], dtype=float)
            else:
                self._vertices = np.array([[vmin], [vmax]], dtype=float)
            return

        # Need at least q+1 points for a full-dimensional hull
        if len(V) < q + 1:
            self._vertices = self._prune_vertices_with_recession_cone(V, cone, tol)
            return

        try:
            hull = ConvexHull(V)
            hull_vertices = V[hull.vertices]
            self._vertices = self._prune_vertices_with_recession_cone(hull_vertices, cone, tol)
        except Exception:
            # Keep unique points
            self._vertices = self._prune_vertices_with_recession_cone(V, cone, tol)

    def _prune_vertices_with_recession_cone(
        self,
        vertices: np.ndarray,
        cone: np.ndarray,
        tol: float,
    ) -> np.ndarray:
        """Remove vertices redundant in ``conv(vertices) + cone``."""
        V = np.asarray(vertices, dtype=float)
        if len(V) <= 1:
            return V

        keep = np.ones(len(V), dtype=bool)
        for i in range(len(V)):
            others = np.delete(V, i, axis=0)
            if len(others) == 0:
                continue

            lam = cp.Variable(len(others), nonneg=True)
            constraints: list[Constraint] = [
                cast(Constraint, cp.sum(lam) == 1)
            ]
            image_point = others.T @ lam

            if cone.size > 0:
                mu = cp.Variable(cone.shape[0], nonneg=True)
                image_point = image_point + cone.T @ mu

            resid = cp.norm(image_point - V[i], 2)
            prob = cp.Problem(cp.Minimize(resid), constraints)

            best_resid = np.inf
            for solver in ("CLARABEL", "SCS"):
                solve_kwargs = {}
                if solver == "SCS":
                    solve_kwargs = {
                        "eps": 1e-6,
                        "max_iters": 100000,
                        "acceleration_lookback": 50,
                    }
                try:
                    prob.solve(solver=solver, warm_start=True, **solve_kwargs)
                except Exception:
                    continue

                if prob.status not in ("optimal", "optimal_inaccurate") or prob.value is None:
                    continue

                resid_val = float(np.asarray(prob.value).item())
                if not np.isfinite(resid_val):
                    continue

                if resid_val < best_resid:
                    best_resid = resid_val

                if prob.status == "optimal" and resid_val <= 10.0 * tol:
                    break

            if np.isfinite(best_resid) and best_resid <= 10.0 * tol:
                keep[i] = False

        pruned = V[keep]
        if len(pruned) == 0:
            # Keep one anchor point to avoid an empty V-representation.
            return V[[0]]
        return pruned


    def _v_to_normals(self) -> Optional[np.ndarray]:
        """Compute outer normals of the polyhedron P + recc.

        Candidate facet normals are taken from ``conv(vertices)`` and then
        filtered by the recession cone condition ``n^T r <= 0`` for all
        generators ``r`` of ``recc`` (facet normals of P + recc are in polar of recc).
        """
        if self._vertices is None:
            self._normals = self._cone_facet_normals.copy()
            return self._normals

        q = self._vertices.shape[1]
        if q != self._graph.q:
            raise ValueError(
                f"vertices must have dimension q={self._graph.q}, but got {q}."
            )
        cone = self.recession_cone_generators
        tol = 1e-9

        def _normal_compatible_with_recc(n: np.ndarray) -> bool:
            if cone.size == 0:
                return True
            return bool(np.all(cone @ n <= tol))

        def _merge_with_cone_normals(a: np.ndarray) -> np.ndarray:
            if a.size == 0:
                return self._cone_facet_normals.copy()
            if self._cone_facet_normals.size == 0:
                return self._deduplicate_rows(self._normalize_rows(a))
            merged = np.vstack([a, self._cone_facet_normals])
            return self._deduplicate_rows(self._normalize_rows(merged))

        # special case q=1: approximation is an interval, normals are -1 and +1
        if q == 1:
            candidates = np.array([[-1.0], [1.0]])
            mask = [
                _normal_compatible_with_recc(candidates[i, :])
                for i in range(candidates.shape[0])
            ]
            self._normals = _merge_with_cone_normals(
                candidates[np.asarray(mask, dtype=bool)]
            )
            return self._normals

        if len(self._vertices) < q + 1:
            # Not enough points for a bounded-hull facet set yet, but keep the precomputed cone-facet normals in place
            self._normals = self._cone_facet_normals.copy()
            return self._normals
        

        hull = ConvexHull(self._vertices)
        # equations: [n_1, ..., n_q, d]  with  ||n|| = 1  and  n^T y + d <= 0
        # first q entries ->> unit outer normals
        normals = hull.equations[:, :-1]

        if cone.size > 0:
            mask = [
                _normal_compatible_with_recc(normals[i, :])
                for i in range(normals.shape[0])
            ]
            normals = normals[np.asarray(mask, dtype=bool)]

        normals = _merge_with_cone_normals(normals)

        self._normals = normals
        return normals

    def _compute_cone_facet_normals_once(self) -> np.ndarray:
        """Compute facet normals of the recession cone once from generators.

        The result is cached and always included in ``self._normals``.
        """
        q = self._graph.q
        cone = self.recession_cone_generators
        tol = 1e-9

        if cone.size == 0:
            return np.empty((0, q), dtype=float)

        def _compatible(n: np.ndarray) -> bool:
            return bool(np.all(cone @ n <= tol))

        if q == 1:
            rays = cone[:, 0]
            has_pos = bool(np.any(rays > tol))
            has_neg = bool(np.any(rays < -tol))
            if has_pos and has_neg:
                return np.empty((0, 1), dtype=float)
            if has_pos:
                return np.array([[-1.0]])
            if has_neg:
                return np.array([[1.0]])
            return np.array([[-1.0], [1.0]])

        norms = np.linalg.norm(cone, axis=1)
        keep = norms > tol
        if not np.any(keep):
            return np.empty((0, q), dtype=float)
        unit_rays = cone[keep] / norms[keep][:, np.newaxis]
        points = np.vstack([np.zeros((1, q)), unit_rays])

        try:
            hull = ConvexHull(points)
        except Exception:
            return np.empty((0, q), dtype=float)

        candidates = []
        for row in hull.equations:
            n = row[:-1]
            d = row[-1]
            if abs(d) > 1e-7:
                continue
            if _compatible(n):
                candidates.append(n)
            elif _compatible(-n):
                candidates.append(-n)

        if not candidates:
            return np.empty((0, q), dtype=float)

        arr = np.asarray(candidates, dtype=float)
        return self._deduplicate_rows(self._normalize_rows(arr))

    @staticmethod
    def _normalize_rows(a: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Normalize row vectors and drop (near-)zero rows."""
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
        """Deduplicate rows by rounded comparison."""
        if a.size == 0:
            return a
        rounded = np.round(a, decimals=decimals)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        return a[np.sort(idx)]
    
    def updateFeasPoint(self, x: np.ndarray) -> None:
        """Update the feasible point associated with this approximation.
        Called on and managed by the approximation algorithm in csop.py.

        Parameters
        ----------
        x : np.ndarray, shape (n,)
            New feasible point x ∈ dom F.
        """
        self._x = x

    def __repr__(self) -> str:
        return (
            f"  vertices=\n{self._vertices},\n"
            f"  normals=\n{self._normals})"
        )


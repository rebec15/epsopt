from __future__ import annotations

from typing import Any, List, Optional, cast

import numpy as np
from scipy.spatial import ConvexHull

from .csop import CSOP
from .approximation import Approximation
from .convex_set import ConvexSet
from .graph import Graph


class ConvexSetApproximator:
    """Wrapper for approximating convex sets with CSOP.

    This class accepts a convex set in CVXPY DCP format.

    Parameters
    ----------
    set : Convex Set in DCP format
       Graph description of the convex set-valued mapping in DCP format.
    solver : str, optional
        CVXPY solver name used by CSOP.
    """

    solver: Optional[str]
    _csop: Optional[CSOP]
    _graph: Optional[Graph]
    _dirty: bool
    _set : ConvexSet
    _set_signature: Optional[tuple[Any, ...]]

    def __init__(
        self,
        convex_set: Optional[ConvexSet] = None,
        *,
        solver: Optional[str] = None,
        **kwargs,
    ) -> None:
        legacy_set = kwargs.pop("set", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")

        if convex_set is None:
            convex_set = legacy_set
        elif legacy_set is not None:
            raise ValueError("Pass either 'convex_set' or legacy 'set', not both.")

        if convex_set is None:
            raise ValueError("convex_set is required.")

        self.solver = solver
        self._csop = None
        self._graph = None
        self._dirty = True
        self._set = convex_set
        self._set_signature = self._snapshot_set_signature()

    def _snapshot_set_signature(self) -> tuple[Any, ...]:
        """Build a signature for detecting in-place mutations of the source set."""
        set_ref = self._set
        fn_ids = tuple(id(fn) for fn in set_ref._constraint_fns)
        recc = tuple(
            tuple(float(v) for v in generator)
            for generator in set_ref.recession_cone_generators
        )
        return (
            set_ref.n,
            set_ref.q,
            set_ref.m,
            set_ref.name,
            fn_ids,
            recc,
        )

    def _refresh_cache_state(self) -> None:
        """Invalidate derived objects when the underlying set definition changed."""
        current = self._snapshot_set_signature()
        if self._set_signature != current:
            self.invalidate_cache()
            self._set_signature = current

    def invalidate_cache(self) -> None:
        """Force rebuilding Graph/CSOP on next access."""
        self._graph = None
        self._csop = None
        self._dirty = True

    def getGraph(self) -> Graph:
        """Lift the convex set S to a graph of a constant set-valued map.

        The returned graph represents
        {(0, x, y) in R^(1 + S.n + S.q) : (x,y) in S}
        with a dummy decision variable x in R fixed to 0 -> dom F = {0}.
        """
        self._refresh_cache_state()

        if self._graph is not None and not self._dirty:
            return self._graph

        set_ref = self._set
        y_dim = set_ref.n + set_ref.q

        def lifted_constraints(x_graph, y_graph, z_graph):
            constraints = [x_graph[0] == 0]
            x_set = y_graph[:set_ref.n]
            y_set = y_graph[set_ref.n:] if set_ref.q > 0 else None
            for fn in set_ref._constraint_fns:
                constraints += fn(x_set, y_set, z_graph)
            return constraints

        self._graph = Graph(
            n=1,
            q=y_dim,
            m=set_ref.m,
            name=f"GraphOf_{set_ref.name}",
            constraint_fns=[lifted_constraints],
            recession_cone_generators=set_ref.recession_cone_generators,
        )
        self._dirty = False
        return self._graph

    def approximate(self, y: Optional[np.ndarray] = None, eps: Optional[float] = None) -> dict:
        """Compute an epsilon-optimizer and return a polyhedral approximation.

        If ``y`` is omitted, a feasible reference point is selected
        automatically from the graph.
        """
        if eps is None:
            raise ValueError("eps must be provided and must satisfy eps > 0.")
        csop = self._get_csop()
        y_arg = None if y is None else np.asarray(y, dtype=float)
        return csop.computeEpsOptimizer(y=y_arg, eps=eps)

    @staticmethod
    def _normalized_recession_rays(generators: np.ndarray, dim: int) -> np.ndarray:
        """Return normalized recession directions with expected dimension."""
        G = np.asarray(generators, dtype=float)
        if G.size == 0:
            return np.empty((0, dim), dtype=float)
        if G.ndim != 2 or G.shape[1] != dim:
            return np.empty((0, dim), dtype=float)
        norms = np.linalg.norm(G, axis=1)
        keep = norms > 1e-12
        if not np.any(keep):
            return np.empty((0, dim), dtype=float)
        return G[keep] / norms[keep][:, np.newaxis]

    @staticmethod
    def _plot_recession_rays_2d(
        ax,
        *,
        anchors: np.ndarray,
        rays: np.ndarray,
        span: np.ndarray,
        color: str,
        label: Optional[str] = None,
    ) -> None:
        """Draw 2D recession rays for unbounded approximations."""
        if anchors.size == 0 or rays.size == 0:
            return
        ray_len = 0.35 * float(np.linalg.norm(span))
        head_w = 0.02 * float(max(span[0], span[1]))
        first = True
        for anchor in anchors:
            for r in rays:
                ax.arrow(
                    float(anchor[0]),
                    float(anchor[1]),
                    float(ray_len * r[0]),
                    float(ray_len * r[1]),
                    width=0.0,
                    head_width=head_w,
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
    def _compute_viewport_2d(
        vertices: np.ndarray,
        generators: np.ndarray,
        *,
        pad_ratio: float = 0.08,
        ray_scale: float = 2.0,
    ) -> tuple[float, float, float, float]:
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

    @classmethod
    def _compute_cone_facet_normals_2d(
        cls,
        generators: np.ndarray,
        tol: float = 1e-9,
    ) -> np.ndarray:
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
        ax,
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
        ax,
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

    @staticmethod
    def _plot_recession_rays_3d(
        ax3,
        *,
        anchors: np.ndarray,
        rays: np.ndarray,
        span: np.ndarray,
        color: str,
        label: Optional[str] = None,
    ) -> None:
        """Draw 3D recession rays for unbounded approximations."""
        if anchors.size == 0 or rays.size == 0:
            return
        ray_len = 0.35 * float(np.linalg.norm(span))
        first = True
        for anchor in anchors:
            for r in rays:
                ax3.quiver(
                    float(anchor[0]),
                    float(anchor[1]),
                    float(anchor[2]),
                    float(ray_len * r[0]),
                    float(ray_len * r[1]),
                    float(ray_len * r[2]),
                    color=color,
                    alpha=0.7,
                    linewidth=1.0,
                    linestyle="--",
                    arrow_length_ratio=0.15,
                    label=label if first else None,
                )
                first = False

    def plot_approximation(
        self,
        approximation: Optional[Approximation] = None,
        *,
        y_choice: Optional[np.ndarray] = None,
        eps: Optional[float] = None,
        set_description: Optional[str] = None,
        ax=None,
        approx_color: str = "tomato",
        approx_alpha: float = 0.25,
        approx_edge_color: str = "tomato",
        approx_vertex_color: str = "tomato",
        approx_vertex_size: float = 0.0,
    ):
        """Plot approximation vertices/hull for image dimension 2 or 3.

        Parameters
        ----------
        approximation : Approximation, optional
            Approximation to plot. If omitted, uses the last result from CSOP.
        y_choice : np.ndarray, optional
            Optional reference point shown as black marker.
        eps : float, optional
            Epsilon value used by the eps-optimizer algorithm.
        set_description : str, optional
            Text description for S in the title, e.g. "{x in R^2 : ||x||_2 <= 1}".
        ax : matplotlib axis, optional
            Target axis. If omitted, a new figure/axis is created.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if approximation is None:
            csop = self._get_csop()
            if not csop.eps_optimizers:
                raise RuntimeError(
                    "No approximation available. Run approximate(...) first or pass approximation explicitly."
                )
            approximation = csop.eps_optimizers[-1]["approx"]

        if approximation is None:
            raise RuntimeError("Approximation is not available.")

        approx = approximation

        vertices = approx.vertices
        if vertices is None or len(vertices) == 0:
            raise RuntimeError("Approximation has no vertices to plot.")

        verts = np.asarray(vertices, dtype=float)
        dim = verts.shape[1]
        if dim not in (2, 3):
            raise ValueError(
                f"plot_approximation is only supported for image dimension 2 or 3, got {dim}."
            )

        if set_description is None:
            set_description = (
                f"(x,y) in R^{self._set.n + self._set.q}, "
                f"exists z in R^{self._set.m}, "
                f"satisfying constraints of {self._set.name}"
            )
        eps_text = "?" if eps is None else f"{eps:g}"
        title = (
            f"Inner Approximation of set S={set_description} "
            f"using the eps-optimizer algorithm with precision eps = {eps_text}"
        )

        x_opt = approx.feasiblePoint
        x_str = np.array2string(np.asarray(x_opt), precision=3, separator=", ", suppress_small=True)

        if dim == 2:
            if ax is None:
                _, ax = plt.subplots()

            approx_label = rf"Approximation of $F(x)$ with x={x_str}"
            recc = np.asarray(approx.recession_cone_generators, dtype=float)
            has_unbounded_recession = (
                recc.ndim == 2
                and recc.shape[1] == 2
                and recc.size > 0
                and len(verts) >= 2
            )

            if has_unbounded_recession:
                viewport = self._compute_viewport_2d(verts, recc)
                self._draw_unbounded_set_fill_2d(
                    ax,
                    vertices=verts,
                    generators=recc,
                    fill_color=approx_color,
                    fill_alpha=approx_alpha,
                    edge_color=approx_edge_color,
                    label=approx_label,
                    viewport=viewport,
                )
                ax.set_xlim(viewport[0], viewport[1])
                ax.set_ylim(viewport[2], viewport[3])
            elif len(verts) >= 3:
                hull = ConvexHull(verts)
                hull_pts = verts[hull.vertices]
                poly = np.vstack([hull_pts, hull_pts[0]])
                ax.fill(
                    poly[:, 0],
                    poly[:, 1],
                    color=approx_color,
                    alpha=approx_alpha,
                    label=approx_label,
                )
                ax.plot(poly[:, 0], poly[:, 1], color=approx_edge_color, linewidth=1.5)
            elif len(verts) == 2:
                ax.plot(
                    verts[:, 0],
                    verts[:, 1],
                    color=approx_edge_color,
                    linewidth=1.5,
                    label=approx_label,
                )
            else:
                ax.scatter(
                    verts[:, 0],
                    verts[:, 1],
                    color=approx_vertex_color,
                    s=max(20.0, approx_vertex_size),
                    zorder=5,
                    label=approx_label,
                )

            ax.scatter(
                verts[:, 0],
                verts[:, 1],
                color=approx_vertex_color,
                s=approx_vertex_size,
                zorder=5,
            )

            if y_choice is not None:
                y_arr = np.asarray(y_choice, dtype=float).ravel()
                if y_arr.shape == (2,):
                    ax.scatter(
                        y_arr[0],
                        y_arr[1],
                        color="black",
                        s=max(30.0, approx_vertex_size * 1.2),
                        marker="x",
                        label=r"$y_0$",
                        zorder=6,
                    )

            ax.set_xlabel(r"$s_1$")
            ax.set_ylabel(r"$s_2$")
            ax.set_title(title)
            ax.legend()
            ax.set_aspect("equal", adjustable="datalim")
            return ax

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        ax3 = cast(Any, ax)
        hull_pts = verts

        if len(verts) >= 4:
            hull = ConvexHull(verts)
            hull_pts = verts[hull.vertices]
            faces = [verts[simplex] for simplex in hull.simplices]
            poly3d = Poly3DCollection(
                faces,
                facecolors=approx_color,
                edgecolors=approx_edge_color,
                alpha=approx_alpha,
                linewidths=0.8,
            )
            ax3.add_collection3d(poly3d)

        ax3.scatter(
            verts[:, 0],
            verts[:, 1],
            zs=verts[:, 2],
            color=approx_vertex_color,
            s=approx_vertex_size,
            label=rf"Approximation of $F(x)$ with x={x_str}",
            depthshade=True,
        )

        if y_choice is not None:
            y_arr = np.asarray(y_choice, dtype=float).ravel()
            if y_arr.shape == (3,):
                ax3.scatter(
                    y_arr[0],
                    y_arr[1],
                    zs=y_arr[2],
                    color="black",
                    s=max(30.0, approx_vertex_size * 1.2),
                    marker="x",
                    label=r"$y_0$",
                    depthshade=False,
                )

        recc = np.asarray(approx.recession_cone_generators, dtype=float)
        rays3 = self._normalized_recession_rays(recc, dim=3)
        if rays3.size > 0 and len(hull_pts) > 0:
            span_now = np.maximum(verts.max(axis=0) - verts.min(axis=0), 1e-8)
            self._plot_recession_rays_3d(
                ax3,
                anchors=hull_pts,
                rays=rays3,
                span=span_now,
                color=approx_edge_color,
                label="recession rays",
            )

        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        if rays3.size > 0 and len(hull_pts) > 0:
            ray_len = 0.35 * float(np.linalg.norm(np.maximum(maxs - mins, 1e-8)))
            endpoints = hull_pts[:, np.newaxis, :] + ray_len * rays3[np.newaxis, :, :]
            end_flat = endpoints.reshape(-1, 3)
            mins = np.minimum(mins, end_flat.min(axis=0))
            maxs = np.maximum(maxs, end_flat.max(axis=0))
        span = np.maximum(maxs - mins, 1e-8)
        ax.set_xlim(mins[0] - 0.05 * span[0], maxs[0] + 0.05 * span[0])
        ax.set_ylim(mins[1] - 0.05 * span[1], maxs[1] + 0.05 * span[1])
        ax3.set_zlim(mins[2] - 0.05 * span[2], maxs[2] + 0.05 * span[2])
        ax.set_xlabel(r"$s_1$")
        ax.set_ylabel(r"$s_2$")
        ax3.set_zlabel(r"$s_3$")
        ax.set_title(title)
        ax.legend()
        return ax

    def _get_csop(self) -> CSOP:
        graph = self.getGraph()
        if self._csop is None or self._csop.graph is not graph:
            self._csop = CSOP(graph, solver=self.solver)
        return self._csop

    def __repr__(self) -> str:
        return (
            f"ConvexSetApproximator(set={self._set.name!r}, "
            f"solver={self.solver!r})"
        )

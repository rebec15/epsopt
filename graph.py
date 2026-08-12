from __future__ import annotations
from typing import Callable, List, Optional, Sequence
import cvxpy as cp
import numpy as np

try:
    import cdd  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    cdd = None

from .convex_set import ConvexSet


class Graph(ConvexSet):
    """ graph F of a convex set-valued function F

    Parameters
    ----------
    n : int
        dimension of decision space (x ∈ R^n).
    q : int
        dimension of image space (y ∈ R^q).
    m : int
        dimension of projection variables (z ∈ R^m)
        Note: projection variables z are necessary e.g. when the graph is a spectrahedral shadow of the form gr F = {(x,y)∈ R^(nxq): ∃ z ∈ R^m s.t. A0 + Σ_i x_i*Ai + Σ_j y_j*Bj + Σ_k z_k*Ck ⪰ 0}

    _constraint_fns: List[Callable] : constraint functions defining the graph
    name : str, optional
    constraint_fns : List[Callable], optional
        Initial constraint-building functions to register.
    recession_cone_generators : Sequence[Sequence[float]], optional
        Generator vectors of a polyhedral recession cone of the common recession cone of the image sets F(x).
        Since these are directions in value space ``y``, each generator must
        have length ``q``.
    """

    # instance attributes
    q: int

    def __init__(
        self,
        n: int,
        q: int,
        m: int = 0,
        name: str = "Graph",
        constraint_fns: Optional[List[Callable]] = None,
        recession_cone_generators: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if q < 1:
            raise ValueError(f"q must be >= 1, but was set to {q}.")

        super().__init__(
            n=n,
            q=q,
            m=m,
            name=name,
            constraint_fns=constraint_fns,
            recession_cone_generators=recession_cone_generators,
        )

    def is_dcp(self) -> bool:
        """Check whether the registered constraints satisfy CVXPY's DCP
        (Disciplined Convex Programming) rules, i.e. whether CVXPY can solve
        optimization problems over this graph.

        Creates temporary dummy variables, calls all registered constraint
        functions, and runs CVXPY's built-in DCP check on the resulting
        problem.

        Returns
        -------
        bool
            True if every constraint is DCP-compliant, False otherwise
            (including if no constraint functions have been registered or if
            building the constraints raises an exception).
        """
        if not self._constraint_fns:
            return False
        x = cp.Variable(self.n, name="_dcp_x")
        y = cp.Variable(self.q, name="_dcp_y")
        z = cp.Variable(self.m, name="_dcp_z") if self.m > 0 else None
        try:
            constraints = self.make_constraints(x, y, z)
            prob = cp.Problem(cp.Minimize(0), constraints)
            return prob.is_dcp()
        except Exception:
            return False

    def find_feasible_point(
        self,
        *,
        solver: Optional[str] = None,
        warm_start: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return one feasible pair ``(x, y)`` with ``(x, y) in gr F``.

        This method certifies that ``img F`` is nonempty by solving a pure
        feasibility problem over the graph constraints.

        Parameters
        ----------
        solver : str, optional
            CVXPY solver to use for the feasibility check.
        warm_start : bool, optional
            Forwarded to CVXPY solve(). Defaults to True.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A feasible decision-image pair ``(x, y)``.

        Raises
        ------
        RuntimeError
            If the graph is infeasible (empty domain/image) or if no reliable
            feasibility status can be obtained.
        """
        if not self._constraint_fns:
            raise RuntimeError(
                "Graph has no constraints. Please add constraints via add_constraint_fn()."
            )

        x = cp.Variable(self.n, name="_feas_x")
        y = cp.Variable(self.q, name="_feas_y")
        z = cp.Variable(self.m, name="_feas_z") if self.m > 0 else None

        try:
            constraints = self.make_constraints(x, y, z)
        except Exception as exc:
            raise RuntimeError(
                "Could not build graph constraints for feasibility check."
            ) from exc

        prob = cp.Problem(cp.Minimize(0), constraints)
        status = self._solve_problem_with_fallback(
            prob,
            solver=solver,
            warm_start=warm_start,
            accepted_statuses=(
                "optimal",
                "optimal_inaccurate",
                "infeasible",
                "infeasible_inaccurate",
            ),
        )

        if status in ("infeasible", "infeasible_inaccurate"):
            raise RuntimeError(
                "Graph is infeasible: dom F is empty (equivalently, img F is empty)."
            )
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                "Could not certify graph feasibility. "
                f"Solver ended with status: {status}."
            )
        if x.value is None or y.value is None:
            raise RuntimeError(
                "Feasibility solve reported optimal status but returned no variable values."
            )

        return (
            np.asarray(x.value, dtype=float).ravel(),
            np.asarray(y.value, dtype=float).ravel(),
        )

    def has_nonempty_image(
        self,
        *,
        solver: Optional[str] = None,
        warm_start: bool = True,
    ) -> bool:
        """Return True iff the graph has at least one feasible pair ``(x, y)``."""
        try:
            self.find_feasible_point(solver=solver, warm_start=warm_start)
            return True
        except RuntimeError:
            return False

    def is_bounded(
        self,
        *,
        solver: Optional[str] = None,
        assume_feasible: bool = False,
    ) -> bool:
        """Check whether img F is bounded modulo the stored recession cone.

        Let K = cone(self.recession_cone_generators). We compute a generating
        set of the polar cone

            K° = {d in R^q : d^T k <= 0 for all k in K}.

        For each polar generator d, we solve

            max d^T y  s.t.  y in img F

        and return False as soon as one problem is unbounded or cannot be
        solved reliably. If these problems are bounded for all generators, they are bounded for all directions of K°. Assume the problem is bounded for all generators. Consider the solutions Y= y^1,...,y^q for all generators. Then img F is bounded given the bounded set B:= aff(conv Y) ∩ img F since img F = B+C.

        Parameters
        ----------
        solver : str, optional
            CVXPY solver to use for boundedness checks.
        assume_feasible : bool, optional
            If True, skip the feasibility pre-check for ``img F`` and assume
            feasibility has already been established externally.

        Returns
        -------
        bool
            True if all support problems over polar generators are finite.
            False if any support problem is unbounded, infeasible, solver
            status is inconclusive, or building/solving constraints fails.
        """
        if not self._constraint_fns:
            return False

        gens_raw = getattr(self, "recession_cone_generators", [])
        if not gens_raw:
            G = np.empty((0, self.q), dtype=float)
        else:
            G = np.asarray(gens_raw, dtype=float)
            if G.ndim != 2 or G.shape[1] != self.q:
                return False

        x = cp.Variable(self.n, name="_bnd_x")
        y = cp.Variable(self.q, name="_bnd_y")
        z = cp.Variable(self.m, name="_bnd_z") if self.m > 0 else None
        try:
            constraints = self.make_constraints(x, y, z)
        except Exception:
            return False

        if not assume_feasible:
            # Feasibility pre-check for img F: return false if img F is empty.
            feas_prob = cp.Problem(cp.Minimize(0), constraints)
            feas_status = self._solve_problem_with_fallback(
                feas_prob,
                solver=solver,
                warm_start=True,
            )
            if feas_status not in ("optimal", "optimal_inaccurate"):
                return False

        directions, polar_is_zero = self._polar_cone_generators(
            G,
            solver=solver,
        )
        if directions.size == 0: # C = R^q
            return polar_is_zero

        direction = cp.Parameter(self.q, name="_bnd_direction")
        prob = cp.Problem(cp.Maximize(direction @ y), constraints)

        for d in directions:
            direction.value = d
            status = self._solve_problem_with_fallback(
                prob,
                solver=solver,
                warm_start=True,
            )
            if status in ("unbounded", "unbounded_inaccurate"):
                return False
            if status not in ("optimal", "optimal_inaccurate"):
                return False

        return True

    @staticmethod
    def _solve_problem_with_fallback(
        prob: cp.Problem,
        *,
        solver: Optional[str] = None,
        warm_start: bool = True,
        accepted_statuses: Optional[Sequence[str]] = None,
        **solve_kwargs,
    ) -> Optional[str]:
        """Solve CVXPY problem with optional solver fallback."""
        accepted = set(accepted_statuses) if accepted_statuses is not None else {
            "optimal",
            "optimal_inaccurate",
            "unbounded",
            "unbounded_inaccurate",
            "infeasible",
            "infeasible_inaccurate",
        }

        candidates: List[Optional[str]] = []
        if solver is not None:
            candidates.append(solver)
        candidates.extend([None, "CLARABEL", "SCS"])

        seen = set()
        last_status: Optional[str] = None
        for candidate in candidates:
            key = candidate if candidate is not None else "__default__"
            if key in seen:
                continue
            seen.add(key)
            try:
                if candidate is None:
                    prob.solve(warm_start=warm_start, **solve_kwargs)
                else:
                    prob.solve(solver=candidate, warm_start=warm_start, **solve_kwargs)

                status = prob.status
                last_status = status
                if status in accepted:
                    return status
            except Exception:
                continue
        return last_status

    def _polar_cone_generators(
        self,
        cone_generators: np.ndarray,
        *,
        tol: float = 1e-9,
        solver: Optional[str] = None,
    ) -> tuple[np.ndarray, bool]:
        """Compute generators of K° from generators of K.

        Returns
        -------
        tuple[np.ndarray, bool]
            ``(D, polar_is_zero)`` where rows of ``D`` generate K° and
            ``polar_is_zero`` indicates whether K° = {0}.
        """
        q = self.q
        G = np.asarray(cone_generators, dtype=float)
        if G.size == 0:
            dirs = []
            for j in range(q):
                ej = np.zeros(q, dtype=float)
                ej[j] = 1.0
                dirs.append(ej)
                dirs.append(-ej)
            return np.asarray(dirs, dtype=float), False

        if G.ndim != 2 or G.shape[1] != q:
            raise ValueError(
                "recession_cone_generators must have shape (k, q) "
                f"with q={q}, but got {G.shape}."
            )

        nonzero = np.linalg.norm(G, axis=1) > tol
        G = G[nonzero]
        if G.shape[0] == 0:
            dirs = []
            for j in range(q):
                ej = np.zeros(q, dtype=float)
                ej[j] = 1.0
                dirs.append(ej)
                dirs.append(-ej)
            return np.asarray(dirs, dtype=float), False

        # treat dimension 1 separately
        if q == 1:
            rays = G[:, 0]
            has_pos = bool(np.any(rays > tol))
            has_neg = bool(np.any(rays < -tol))
            if has_pos and has_neg:
                # K = R, hence K° = {0}.
                return np.empty((0, 1), dtype=float), True
            if has_pos:
                return np.array([[-1.0]]), False
            if has_neg:
                return np.array([[1.0]]), False
            return np.array([[-1.0], [1.0]]), False

        if cdd is None:
            raise RuntimeError(
                "cddlib (pycddlib) is required to compute polar cone generators."
            )

        # cddlib uses b + A x >= 0. From G d <= 0 (constraints for polar cone) we get (-G) d >= 0,
        # therefore b = 0 and A = -G.
        H = np.hstack([np.zeros((G.shape[0], 1), dtype=float), -G])

        try:
            if hasattr(cdd, "matrix_from_array"):
                mat = cdd.matrix_from_array(H, rep_type=cdd.RepType.INEQUALITY)
                poly = cdd.polyhedron_from_matrix(mat)
                generators = cdd.copy_generators(poly)
            else:
                mat = cdd.Matrix(H.tolist(), number_type="fraction")
                mat.rep_type = cdd.RepType.INEQUALITY
                poly = cdd.Polyhedron(mat)
                generators = poly.get_generators()
        except Exception as exc:
            raise RuntimeError(
                "Failed to compute polar cone generators via cddlib."
            ) from exc

        rows_obj = getattr(generators, "array", generators)
        rows = np.asarray(rows_obj, dtype=float)
        if rows.size == 0:
            return np.empty((0, q), dtype=float), True
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.shape[1] != q + 1:
            raise RuntimeError(
                "Unexpected cddlib generator format: expected q+1 columns, "
                f"got {rows.shape[1]} for q={q}."
            )

        lin_set = set(int(i) for i in getattr(generators, "lin_set", set()))
        candidates: List[np.ndarray] = []

        for idx_row, row in enumerate(rows):
            t = float(row[0])
            v = np.asarray(row[1:], dtype=float)
            nrm = np.linalg.norm(v)
            if nrm <= tol:
                continue

            # lineality rows represent free directions; add both orientations
            if idx_row in lin_set:
                candidates.append(v)
                candidates.append(-v)
                continue

            # t == 0 marks rays; t == 1 marks vertices (ignored for cones)
            if abs(t) <= tol:
                candidates.append(v)

        if not candidates:
            return np.empty((0, q), dtype=float), True

        D = np.asarray(candidates, dtype=float)
        D_norm = np.linalg.norm(D, axis=1)
        keep = D_norm > tol
        D = D[keep] / D_norm[keep][:, np.newaxis]
        rounded = np.round(D, decimals=10)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        D = D[np.sort(idx)]
        return D, False

    def validate(
        self,
        check_dcp: bool = False,
        check_bounded: bool = False,
        check_recession_cone: bool = False,
        solver: Optional[str] = None,
    ) -> None:
        """Check that the graph is properly defined.

        Parameters
        ----------
        check_dcp : bool, optional
            If True, also verify that all constraints satisfy CVXPY's DCP
            rules.  Raises RuntimeError when the check fails.
            Default is False (skip the DCP check).
        check_bounded : bool, optional
            If True, also verify that graph F is bounded.
            Raises RuntimeError when the check fails.
            Default is False (skip the boundedness check).
        check_recession_cone : bool, optional
            If True, also verify that either the recession cone has
            nonempty interior or it is the trivial cone {0}.
            Raises RuntimeError when the check fails.
            Default is False (skip this check).
        solver : str, optional
            CVXPY solver used for feasibility and boundedness checks.

        Note: For this package, the algorithm in csop.py requires graph F to
        be bounded (compact image sets F(x)).
        """
        if not self._constraint_fns:
            raise RuntimeError(
                "Graph has no constraints. Please add constraints via add_constraint_fn()."
            )
        if check_dcp:
            if not self.is_dcp():
                raise RuntimeError(
                    "Graph constraints are not DCP-compliant. "
                    "CVXPY cannot solve optimization problems over this graph as defined. "
                )
            print("Graph constraints are DCP-compliant.")
        if check_recession_cone:
            try:
                self._ensure_recession_cone_has_nonempty_interior(
                    self.recession_cone_generators,
                    tol=1e-10,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Recession cone condition failed: expected either int(recc(F)) "
                    "to be nonempty or recc(F) = {0}."
                ) from exc
            print("Recession cone has full dimension or is {0}.")
        if check_bounded:
            if not self.has_nonempty_image(solver=solver):
                raise RuntimeError(
                    "Graph is infeasible: dom F is empty (equivalently, img F is empty). "
                    "No pair (x, y) satisfies the graph constraints."
                )
            if not self.is_bounded(solver=solver, assume_feasible=True):
                raise RuntimeError(
                    "Graph graph F is not bounded. "
                    "The algorithm requires that img F ⊆ B + 0^+F(x) for some compact B. "
                    "Add finite upper and lower bounds y in your constraints to run the algorithm for a bounded section of the graph."
                )
            print("Mapping F is bounded.")

    def _recession_generator_dim(self) -> int:
        """For graphs of set-valued maps, value-space is y in R^q."""
        return self.q

    def __repr__(self) -> str:
        return (
            f"Graph(name={self.name!r}, n={self.n}, q={self.q}, "
            f"#constraint_fns={len(self._constraint_fns)}, "
            f"#recession_generators={len(self.recession_cone_generators)})"
        )

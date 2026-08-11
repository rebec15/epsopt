from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import cvxpy as cp
import numpy as np


class ConvexSet:
    """Base class for convex sets over variables (x, y, z) defined via dcp-compliant constraints.

    Parameters
    ----------
    n : int
        Dimension of decision variable x in R^n.
    q : int
        Dimension of image variable y in R^q. If q=0, no y variable exists.
    m : int
        Dimension of projection variable z in R^m.
    name : str, optional
        Name.
    constraint_fns : list[Callable], optional
        Initial constraint builders.
    recession_cone_generators : Sequence[Sequence[float]], optional
        Generator vectors of a polyhedral recession of the convex set.
        For ``ConvexSet``, each generator must have length ``n + q``.
    """

    n: int
    q: int
    m: int
    name: str
    x: cp.Variable
    y: Optional[cp.Variable]
    z: Optional[cp.Variable]
    _constraint_fns: List[Callable]
    recession_cone_generators: List[List[float]]

    def __init__(
        self,
        n: int,
        q: int,
        m: int = 0,
        name: str = "ConvexSet",
        constraint_fns: Optional[List[Callable]] = None,
        recession_cone_generators: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, but was set to {n}.")
        if q < 0:
            raise ValueError(f"q must be >= 0, but was set to {q}.")
        if m < 0:
            raise ValueError(f"m must be >= 0, but was set to {m}.")

        self.n = n
        self.q = q
        self.m = m
        self.name = name

        self.x = cp.Variable(n, name="x")
        self.y = None
        if self.q > 0:
            self.y = cp.Variable(q, name="y")

        self.z = None
        if self.m > 0:
            self.z = cp.Variable(m, name="z")

        self._constraint_fns = list(constraint_fns) if constraint_fns is not None else []
        self.recession_cone_generators = []

        if recession_cone_generators is not None:
            self.set_recession_cone_generators(recession_cone_generators)
            self.remove_redundant_recession_cone_generators()

    def set_recession_cone_generators(
        self, generators: Sequence[Sequence[float]]
    ) -> None:
        """Set generator vectors of a polyhedral recession cone.

        Each generator must have length ``self._recession_generator_dim()``.
        """
        expected_dim = self._recession_generator_dim()
        listGen: List[List[float]] = []
        for g in generators:
            g_list = list(g)
            if len(g_list) != expected_dim:
                raise ValueError(
                    "Each recession cone generator must have length "
                    f"{expected_dim}, but got {len(g_list)}."
                )
            listGen.append(g_list)
        self._ensure_recession_cone_has_nonempty_interior(listGen)
        self.recession_cone_generators = listGen

    def add_recession_cone_generator(self, generator: Sequence[float]) -> None:
        """Add one generator vector to the polyhedral recession cone."""
        expected_dim = self._recession_generator_dim()
        g_list = list(generator)
        if len(g_list) != expected_dim:
            raise ValueError(
                "Recession cone generator must have length "
                f"{expected_dim}, but got {len(g_list)}."
            )
        self.recession_cone_generators.append(g_list)

    def remove_redundant_recession_cone_generators(
        self,
        *,
        tol: float = 1e-9,
        solver: Optional[str] = None,
    ) -> List[List[float]]:
        """Remove redundant recession-cone generators in-place.

        A generator is considered redundant if it can be represented (up to
        tolerance ``tol``) as a nonnegative linear combination of the other
        generators.

        Parameters
        ----------
        tol : float, optional
            Absolute feasibility tolerance used in the conic representation
            checks. Must be nonnegative.
        solver : str, optional
            CVXPY solver name used for redundancy checks.

        Returns
        -------
        list[list[float]]
            The reduced generator list that remains stored in
            ``self.recession_cone_generators``.
        """
        if tol < 0:
            raise ValueError(f"tol must be >= 0, but was set to {tol}.")

        if not self.recession_cone_generators:
            return self.recession_cone_generators

        G = np.asarray(self.recession_cone_generators, dtype=float)
        if G.ndim != 2:
            raise ValueError(
                "recession_cone_generators must be a 2D array-like object."
            )

        # Drop near-zero vectors first; they do not contribute to the cone.
        keep_nonzero = np.linalg.norm(G, axis=1) > tol
        G = G[keep_nonzero]

        def _to_list_of_float_rows(arr: np.ndarray) -> List[List[float]]:
            return [list(map(float, row)) for row in arr]

        if G.shape[0] <= 1:
            self.recession_cone_generators = _to_list_of_float_rows(G)
            return self.recession_cone_generators

        keep = np.ones(G.shape[0], dtype=bool)

        for i in range(G.shape[0]):
            if not keep[i]:
                continue

            others_idx = [j for j in range(G.shape[0]) if keep[j] and j != i]
            if not others_idx:
                continue

            A = G[others_idx].T
            g_i = G[i]
            lam = cp.Variable(len(others_idx), nonneg=True)
            residual = A @ lam - g_i
            prob = cp.Problem(
                cp.Minimize(cp.sum(lam)),
                [residual <= tol, residual >= -tol],
            )

            solved = False
            for candidate_solver in ([solver] if solver else []) + ["CLARABEL", "SCS"]:
                try:
                    prob.solve(solver=candidate_solver)
                    if prob.status in ("optimal", "optimal_inaccurate"):
                        solved = True
                        break
                except Exception:
                    continue

            if solved:
                keep[i] = False

        G_reduced = G[keep]
        self.recession_cone_generators = _to_list_of_float_rows(G_reduced)
        return self.recession_cone_generators

    def _recession_generator_dim(self) -> int:
        """Dimension of one recession-cone generator in this model.

        For generic ``ConvexSet`` objects, value-space is represented by
        coordinates ``(x, y)``, hence dimension ``n + q``.
        """
        return self.n + self.q

    def _ensure_recession_cone_has_nonempty_interior(
        self,
        generators: Sequence[Sequence[float]],
        *,
        tol: float = 1e-10,
    ) -> None:
        """Raise if the cone is neither full-dimensional nor the trivial cone.

        For a polyhedral cone in R^d, nonempty interior is equivalent to
        full-dimensionality in R^d. As a special case, the trivial recession
        cone ``{0}`` is accepted as well (represented by no generators or only
        near-zero generators).
        """
        d = self._recession_generator_dim()
        if d <= 0:
            return

        G = np.asarray(generators, dtype=float)
        if G.size == 0:
            # No generators encode the trivial cone K = {0}.
            return

        if G.ndim != 2 or G.shape[1] != d:
            raise ValueError(
                "recession_cone_generators must have shape (k, d) "
                f"with d={d}, but got {G.shape}."
            )

        nonzero = np.linalg.norm(G, axis=1) > tol
        G = G[nonzero]
        if G.shape[0] == 0:
            # Near-zero generators also encode K = {0}.
            return

        rank = int(np.linalg.matrix_rank(G, tol=tol))
        if rank < d:
            raise ValueError(
                "Recession cone has empty interior"
                f"(rank {rank} < ambient dimension {d})."
            )

    def add_constraint_fn(self, fn: Callable) -> None:
        """Register one constraint-building function.

        The expected signature is ``fn(x, y, z)`` where missing variables are
        passed as ``None`` (i.e. ``y is None`` if ``q == 0`` and
        ``z is None`` if ``m == 0``).
        """
        self._constraint_fns.append(fn)

    def add_constraint_fns(self, fns: List[Callable]) -> None:
        """Register multiple constraint-building functions at once."""
        self._constraint_fns.extend(fns)

    def make_constraints(self, x_var, y_var=None, z_var=None) -> List[cp.Constraint]:
        """Build CVXPY constraints from all registered builders.

        The expected call signature is always ``make_constraints(x, y, z)``.
        Missing variables should be passed as ``None``.
        """
        result: List[cp.Constraint] = []
        for fn in self._constraint_fns:
            result += fn(x_var, y_var, z_var)
        return result

    def __repr__(self) -> str:
        return (
            f"ConvexSet(name={self.name!r}, n={self.n}, q={self.q}, m={self.m}, "
            f"#constraint_fns={len(self._constraint_fns)}, "
            f"#recession_generators={len(self.recession_cone_generators)})"
        )

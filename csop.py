"""
csop.py – Convex Set-valued Optimization Problem (CSOP) and γε-optimizer algorithm.

The main entry point is :class:`CSOP`, which wraps a :class:`~epsopt.graph.Graph` and
exposes :meth:`CSOP.computeEpsOptimizer` to compute a γε-optimizer.

Algorithm overview
------------------
Given a convex set-valued function F: R^n ⇉ R^q with compact graph, a point y ∈ img F and a tolerance ε > 0, the algorithm iteratively refines a polyhedral inner approximation I of F(x) for some γε-optimizer x with y ∈ F(x) and finally returns the approximation and the γε-optimizer by solving two subproblems:

  CP (convex program) – finds a new point in the domain x and candidate point y* by maximising w^T y over (x, y) ∈ graph F subject to vertex-feasibility constraints.
  IP (inner-point) – extends the approximation by computing a boundary point of F(x) in a given direction.

The loop terminates when every outer normal w of the approximation polytope
has been used as a CP weight and the respective normal could not be shifted outwards by more than an ε-distance
"""

from __future__ import annotations
from typing import Callable, List, Optional
import copy
import numpy as np
import cvxpy as cp
from scipy.linalg import null_space

from .graph import Graph
from .cp import CP
from .ip import IP
from .approximation import Approximation

class CSOP:
    """Convex Set-valued Optimization Problem.

    Wraps a :class:`~epsopt.graph.Graph` and provides the γε-optimizer
    algorithm via :meth:`computeEpsOptimizer`.

    Parameters
    ----------
    graph : Graph
        The graph of the set-valued function F.  Must have at least one
        constraint function registered.
    solver : str, optional
        cvxpy solver name (e.g. ``"CLARABEL"``, ``"MOSEK"``).

    Attributes
    ----------
    eps_optimizers : list of dict
        Accumulated results from all :meth:`computeEpsOptimizer` calls.
        Each entry has keys ``"y_choice"``, ``"eps"``, ``"x"``, ``"approx"``,
        ``"gamma_upper_bound"``.
    """

    # instance attributes
    graph: Graph
    solver: Optional[str]
    eps_optimizers: List[dict]          # list of {"y", "eps", "x", "approx"}
    _approximation: Approximation       # current polyhedral approximation
    _ip: IP                             # re-used inner-point subproblem
    _cp: CP                             # current convex subproblem
    x_star: np.ndarray                  # current best x
    y_star: np.ndarray                  # current best y

    def __init__(
        self,
        graph: Graph,
        *,
        solver: Optional[str] = None,
    ) -> None:
        self.graph = graph
        
        # validate if gr F is dcp-compliant and bounded: necessary for computeEpsOptimizer()
        self.graph.validate(check_dcp=True,check_bounded=True)

        self.eps_optimizers: List[dict] = []   # list of {"y", "eps", "x", "approx"}
        self.solver = solver

    def computeEpsOptimizer(
        self, y: np.ndarray, eps: float,
        compute_gamma_upper_bound: bool = False,
        on_iteration: Optional[Callable[[int, Approximation], None]] = None,
    ) -> dict:
        """Compute a γε-optimizer for the CSOP.

        Parameters
        ----------
        y : np.ndarray, shape (q,)
            Reference point in image space.  The algorithm finds x* such that y ∈ F(x*) and x* is an optimizer of the problem defined by graph F, i.e. ∄ x ∈ dom F such that F(x*) ⊊ F(x).
        eps : float
            Approximation tolerance ε > 0.
        compute_gamma_upper_bound : bool, optional
            If True, compute the upper bound ``L`` for ``gamma`` from the
            final normal system by solving SOCPs for all nonempty subsets of
            the final normal set. If False, skip this potentially expensive
            post-processing step.
        on_iteration : callable (int, Approximation) -> None, optional
            Called after each algorithm iteration with the current
            iteration index and a snapshot of the approximation.  Index 0 is
            the initial approximation before the main loop. Used by examples/portfolio_opt.py in order to output a step-by-step visualization of the algorithm.

        Returns
        -------
        dict with keys:
            ``"y_choice"`` – the input reference point *y*.
            ``"eps"``      – the tolerance *eps*.
            ``"x"``        – optimal decision vector x* ∈ R^n.
            ``"approx"``   – :class:`~epsopt.approximation.Approximation` of F(x*).
            ``"gamma_upper_bound"`` – computable upper bound for γ from final normals.
        """
        y = np.asarray(y, dtype=float).ravel()
        if y.shape != (self.graph.q,):
            raise ValueError(
                f"y must have shape ({self.graph.q},) matching graph.q, but has shape {y.shape}."
            )
        if eps <= 0:
            raise ValueError(f"eps must be > 0, but was set to {eps}.")
        delta = min(eps / 4.0, 0.001) # allow for tolerance in solvers of at most eps/4 -> feasibility and duality gap

        # compute initial point
        x = self._computeInitialPoint(y)

        # If 0^+F = R^q, every feasible x is an optimizer because F(x) = R^q.
        # Return immediately with a trivial one-point anchor approximation.
        if self._recession_cone_is_full_space():
            self.x_star = np.asarray(x, dtype=float).ravel()
            self.y_star = y.copy()
            self._approximation = Approximation(self.x_star, self.graph)  # type: ignore[arg-type]
            self._approximation.add_vertex(y)

            if on_iteration is not None:
                on_iteration(0, copy.deepcopy(self._approximation))

            gamma_upper: Optional[float]
            if compute_gamma_upper_bound:
                recession_gens = np.asarray(self.graph.recession_cone_generators, dtype=float)
                polar_gens, _ = self.graph._polar_cone_generators(
                    recession_gens,
                    solver=self.solver,
                )
                gamma_upper = self._compute_gamma_upper_bound(polar_gens)
            else:
                gamma_upper = None

            result = {
                "y_choice": y,
                "eps": eps,
                "x": self.x_star,
                "approx": self._approximation,
                "gamma_upper_bound": gamma_upper,
            }
            self.eps_optimizers.append(result)
            return result

        # Compute initial approximation
        self._computeInitialApproximation(x, y)
        # Keep one CP instance and let CP.solve() rebuild lazily when vertices change.
        self._cp = CP(self.graph, self._approximation, np.ones(self.graph.q))

        if on_iteration is not None:
            on_iteration(0, copy.deepcopy(self._approximation))

        W: set[tuple] = set()
        _iter = 1

        while True:
            N = self._approximation.normals
            if N is None:
                raise RuntimeError("Approximation has no normals. Initial approximation may have too few vertices.")

            # Regular termination: all normals already processed.
            if all(tuple(w.round(15)) in W for w in N):
                break

            # w* = argmax over w in N\W of CP(F, w, I)
            # Be robust against occasional solver outputs with non-finite optval.
            best_w, best_sol, best_val = None, None, -np.inf
            fallback_w, fallback_sol = None, None
            for w in N:
                w_key = tuple(w.round(15)) # compatible with set()
                if w_key in W:
                    continue
                sol = self._cp.solve(
                    w,
                    solver=self.solver,
                    **self._cp_solver_tolerance_kwargs(delta),
                )
                if (
                    fallback_sol is None
                    and sol.get("x") is not None
                    and sol.get("y") is not None
                ):
                    fallback_w, fallback_sol = w, sol

                score = sol.get("optval")
                if score is None or not np.isfinite(score):
                    y_sol = sol.get("y")
                    if y_sol is not None:
                        y_arr = np.asarray(y_sol, dtype=float).ravel()
                        if np.all(np.isfinite(y_arr)) and y_arr.shape == (self.graph.q,):
                            score = float(np.dot(w, y_arr))

                if score is not None and np.isfinite(score) and score > best_val:
                    best_val = score
                    best_w = w
                    best_sol = sol

            if best_w is None and fallback_w is not None and fallback_sol is not None:
                best_w, best_sol = fallback_w, fallback_sol

            if best_w is None or best_sol is None:
                raise RuntimeError("No unused weight vector found in N\\W. Algorithm may have terminated early.")

            # W ← W ∪ {w*}
            W.add(tuple(best_w.round(15)))

            if best_sol["x"] is None or best_sol["y"] is None:
                raise RuntimeError(
                    f"CP subproblem returned no solution for w={best_w} "
                    f"(status: {best_sol['status']}). "
                    "This may indicate a numerical issue with the solver. "
                    "Try a different solver or tighten the graph constraints."
                )

            self.y_star = best_sol["y"]
            self.x_star = best_sol["x"]

            # y_star ∉ I + (ε/2)B  →  update approximation
            if not self._in_eps_approximation(self.y_star, eps/2):
                self._approximation.add_vertex(self.y_star)
                self._approximation.update_feasible_point(self.x_star)
                N = self._approximation.normals
                if on_iteration is not None:
                    on_iteration(_iter, copy.deepcopy(self._approximation))
            
            _iter += 1

            # until N ⊆ W
            assert N is not None
            if all(tuple(w.round(15)) in W for w in N):
                break

        gamma_upper: Optional[float]
        if compute_gamma_upper_bound:
            recession_gens = np.asarray(self.graph.recession_cone_generators, dtype=float)
            polar_gens, _ = self.graph._polar_cone_generators(
                recession_gens,
                solver=self.solver,
            )
            gamma_upper = self._compute_gamma_upper_bound(polar_gens)
        else:
            gamma_upper = None

        # save γε-optimizer to list
        self.eps_optimizers.append({
            "y_choice": y,
            "eps": eps,
            "x": self.x_star,
            "approx": self._approximation, # Note: approximation quality = gamma*eps for some gamma > 0
            "gamma_upper_bound": gamma_upper,
        })

        # return γε-optimizer
        return ({
            "y_choice": y,
            "eps": eps,
            "x": self.x_star,
            "approx": self._approximation,
            "gamma_upper_bound": gamma_upper,
        })

    def _compute_gamma_upper_bound(self, gens: Optional[np.ndarray]) -> Optional[float]:
        """Compute gamma(E) with an SOCP for E being a finite system of generators of the polar cone of the values.

        Solve

            gamma(E) = max 1^T lambda
                s.t. ||E^T lambda||_2 <= 1, lambda >= 0.
        """
        if gens is None:
            return None

        E = np.asarray(gens, dtype=float)
        if E.ndim != 2 or E.shape[0] == 0:
            return None

        # Normalize and deduplicate to improve numerical stability.
        norms = np.linalg.norm(E, axis=1)
        keep = norms > 1e-10
        if not np.any(keep):
            return None
        E = E[keep] / norms[keep][:, np.newaxis]
        rounded = np.round(E, decimals=10)
        _, idx = np.unique(rounded, axis=0, return_index=True)
        E = E[np.sort(idx)]
        m = E.shape[0]
        if m == 0:
            return None

        lam = cp.Variable(m, nonneg=True)
        prob = cp.Problem(cp.Maximize(cp.sum(lam)), [cp.norm(E.T @ lam, 2) <= 1])

        for solver in ([self.solver] if self.solver else []) + ["CLARABEL", "SCS"]:
            try:
                prob.solve(solver=solver)
            except Exception:
                continue

            if prob.status in ("unbounded", "unbounded_inaccurate"):
                return float("inf")

            if prob.status == "optimal" and prob.value is not None:
                val = float(np.asarray(prob.value).item())
                if np.isfinite(val):
                    return val

        return None

    def _cp_solver_tolerance_kwargs(self, delta: float) -> dict:
        """Return CP solver tolerances derived from delta = eps/4.

        Only CP solves use this tolerance budget.
        """
        # Keep tolerances in a numerically safe range.
        gap_tol = float(max(1e-8, min(delta, 1e-3)))
        feas_tol = float(max(1e-9, min(delta, 1e-5)))

        if self.solver is None:
            return {}

        s = self.solver.upper()
        if s == "SCS":
            return {
                "eps": gap_tol,
                "max_iters": 20000,
                "acceleration_lookback": 10,
            }
        if s == "CLARABEL":
            return {
                "tol_gap_abs": gap_tol,
                "tol_gap_rel": gap_tol,
                "tol_feas": feas_tol,
                "tol_infeas_abs": feas_tol,
                "tol_infeas_rel": feas_tol,
                "max_iter": 1000,
            }
        return {}

    def _computeInitialPoint(self, y_val: np.ndarray) -> np.ndarray:
        # y must be a cp.Parameter so that lambda constraints produce cvxpy constraints
        y = cp.Parameter(self.graph.q)
        y.value = y_val
        x = cp.Variable(self.graph.n)
        z = cp.Variable(self.graph.m) if self.graph.m>0 else None
        constraints = self.graph.make_constraints(x, y, z if self.graph.m>0 else None)
        problem = cp.Problem(cp.Maximize(0), constraints)
        status = self.graph._solve_problem_with_fallback(
            problem,
            solver=self.solver,
            warm_start=True,
            accepted_statuses=(
                "optimal",
                "optimal_inaccurate",
                "infeasible",
                "infeasible_inaccurate",
            ),
        )
        if status not in ("optimal", "optimal_inaccurate") or x.value is None:
            raise RuntimeError(
                f"Could not find a feasible initial point for y={y_val} (status: {status}). "
                "The reference point y may lie outside the image of F, or the graph "
                "constraints are infeasible. Check your constraints and the choice of y."
            )
        return np.asarray(x.value, dtype=float).ravel()
    
    def _computeInitialApproximation(self, x: np.ndarray, y: np.ndarray) -> None:
        self._approximation = Approximation(x, self.graph)  # type: ignore

        # Algorithm InitialApproxUpdate: use recession cone G(0) from graph.
        y_star = self._compute_dual_direction_of_recession_cone()
        p = self._compute_interior_seed_point(y)
        directions = self._build_initial_directions(y_star)

        # P starts with {p, y_bar}.
        self._approximation.add_vertex(p)
        self._approximation.add_vertex(y)

        # Reuse one IP instance and update parameter d each call.
        self._ip = IP(self.graph, x, p, directions[0])
        for d in directions:
            sol = self._ip.solve(x, p, d, solver=self.solver)
            if sol["y"] is None:
                raise RuntimeError(
                    f"IP subproblem returned no solution for direction d={d} "
                    f"(status: {sol['status']}). Check that graph F is feasible and that d is outside G(0)."
                )
            self._approximation.add_vertex(sol["y"])

    def _compute_interior_seed_point(self, y_bar: np.ndarray) -> np.ndarray:
        """Compute a practical interior seed point p from y_bar and G(0)=cone(c^1, ..., c^k)
            -> p = y_bar + 1/k sum_k(c^i)
        """
        gens_raw = getattr(self.graph, "recession_cone_generators", [])
        if not gens_raw:
            return y_bar

        G = np.asarray(gens_raw, dtype=float)
        c = np.mean(G, axis=0)
        if np.linalg.norm(c) <= 1e-12:
            c = G[0]
        return y_bar + c

    def _build_initial_directions(self, y_star: np.ndarray) -> List[np.ndarray]:
        """Build directions d^1,...,d^q as in the InitialApproxUpdate algorithm."""
        q = self.graph.q
        norm = np.linalg.norm(y_star)
        if norm <= 1e-12:
            raise ValueError("y_star must be nonzero.")
        d0 = y_star / norm

        # Complete d0 to an orthonormal basis [d0, u1, ..., u_{q-1}].
        U = self._complete_orthonormal_basis(d0)
        u_cols = U[:, 1:] if q > 1 else np.empty((q, 0))

        dirs: List[np.ndarray] = []
        for i in range(q - 1):
            dirs.append(-d0 + u_cols[:, i])
        if q > 1:
            dirs.append(-d0 - np.sum(u_cols, axis=1))
        else:
            dirs.append(-d0)
        return dirs

    def _complete_orthonormal_basis(self, first: np.ndarray) -> np.ndarray:
        """Return a strictly orthonormal basis with first column ``first``.

        Construction:
        1) normalize ``first`` to v,
        2) compute an orthonormal basis of v^perp via scipy.linalg.null_space,
        3) prepend v exactly as first column.
        """
        q = self.graph.q
        v = np.asarray(first, dtype=float).ravel()
        if v.shape != (q,):
            raise ValueError(f"first must have shape ({q},), but got {v.shape}.")

        nrm = np.linalg.norm(v)
        if nrm <= 1e-12:
            raise ValueError("first must be nonzero.")
        v = v / nrm

        if q == 1:
            return v.reshape(1, 1)

        # null_space(v^T) returns a basis of all vectors orthogonal to v.
        Q2 = null_space(v.reshape(1, -1))
        if Q2.shape != (q, q - 1):
            raise RuntimeError(
                "Could not construct full orthonormal complement for first vector. "
                f"Expected shape {(q, q - 1)}, got {Q2.shape}."
            )

        Q = np.column_stack([v, Q2])
        return Q
    
    def _in_eps_approximation(self, y: np.ndarray, eps: float) -> bool:
        """Check if y ∈ I + εB for I = conv(V) + K.

        Here K is built from ``graph.recession_cone_generators`` and is
        interpreted as the common recession cone 0^+F(x) for all feasible x.
        """
        V = self._approximation.vertices
        if V is None:
            return False

        # project y onto conv(V) + K:
        # min ||y - (V^T λ + G^T μ)||
        # s.t. λ ≥ 0, sum(λ)=1, μ ≥ 0
        lam = cp.Variable(len(V), nonneg=True)
        gens_raw = getattr(self.graph, "recession_cone_generators", [])
        if gens_raw:
            G = np.asarray(gens_raw, dtype=float)
            if G.ndim != 2 or G.shape[1] != self.graph.q:
                raise ValueError(
                    "recession_cone_generators must have shape (k, q) "
                    f"with q={self.graph.q}, but got {G.shape}."
                )
            mu = cp.Variable(G.shape[0], nonneg=True)
            image_point = V.T @ lam + G.T @ mu
        else:
            image_point = V.T @ lam

        dist = cp.norm(y - image_point)
        prob = cp.Problem(cp.Minimize(dist), [cp.sum(lam) == 1])  # type: ignore[arg-type]
        for solver in ([self.solver] if self.solver else []) + ["CLARABEL", "SCS"]: # try different solver if one fails
            try:
                prob.solve(solver=solver)
                if prob.value is not None:
                    break
            except Exception:
                continue
        return bool(prob.value is not None and prob.value <= eps)  # type: ignore[operator]

    def _compute_dual_direction_of_recession_cone(self) -> np.ndarray:
        """Compute a nonzero direction y* in G(0)^+ from stored generators.

        Uses SOCPs of the form max a^T y s.t. C y >= 0, ||y||_2 <= 1.
        """
        q = self.graph.q
        gens_raw = getattr(self.graph, "recession_cone_generators", [])
        if not gens_raw:
            vec = np.ones(q, dtype=float)
            return vec / np.linalg.norm(vec)

        C = np.asarray(gens_raw, dtype=float)
        if C.ndim != 2 or C.shape[1] != q:
            raise ValueError(
                "recession_cone_generators must have shape (k, q) "
                f"with q={q}, but got {C.shape}."
            )

        y_var = cp.Variable(q)
        constraints = [cp.norm(y_var, 2) <= 1, C @ y_var >= 0]
        objectives: List[np.ndarray] = [np.ones(q)] + [np.eye(q)[i] for i in range(q)]

        best: Optional[np.ndarray] = None
        best_obj = -np.inf
        for a in objectives:
            prob = cp.Problem(cp.Maximize(a @ y_var), constraints)
            try:
                prob.solve(solver=self.solver)
            except Exception:
                try:
                    prob.solve(solver="CLARABEL")
                except Exception:
                    try:
                        prob.solve(solver="SCS")
                    except Exception:
                        continue

            if y_var.value is None:
                continue
            cand = np.asarray(y_var.value, dtype=float).ravel()
            nrm = np.linalg.norm(cand)
            val = float(a @ cand)
            if nrm > 1e-10 and np.isfinite(val) and val > best_obj:
                best = cand / nrm
                best_obj = val

        if best is None:
            raise RuntimeError(
                "Could not compute a nonzero direction y* in the dual cone G(0)^+. "
                "Please verify recession_cone_generators in the graph."
            )
        return best

    def _recession_cone_is_full_space(self) -> bool:
        """Return True if the stored recession cone equals R^q.

        For a closed convex cone K, this is equivalent to K° = {0}.
        """
        gens_raw = getattr(self.graph, "recession_cone_generators", [])
        if not gens_raw:
            return False

        G = np.asarray(gens_raw, dtype=float)
        if G.ndim != 2 or G.shape[1] != self.graph.q:
            raise ValueError(
                "recession_cone_generators must have shape (k, q) "
                f"with q={self.graph.q}, but got {G.shape}."
            )

        _, polar_is_zero = self.graph._polar_cone_generators(
            G,
            solver=self.solver,
        )
        return bool(polar_is_zero)

    def __repr__(self) -> str:
        return (
            f"CSOP(graph={self.graph.name!r})"
        )
    


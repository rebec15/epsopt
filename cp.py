from __future__ import annotations
from typing import List, Optional
import numpy as np
import cvxpy as cp

from .graph import Graph
from .approximation import Approximation
from ._subproblem import SubProblem


class CP(SubProblem):
    """Subproblem (CP(F,w,I))

    For weight w ∈ R^q and current inner approximation I with vertex set V = {ȳ_1, ..., ȳ_k}:
        max   w^T y
        s.t.  (x, y) ∈ graph F
              ∀ ȳ_i ∈ V : (x, ȳ_i) ∈ graph F  (iff  I ⊆ F(x))
    """

    # instance attributes
    _graph: Graph
    _approximation: Approximation
    _w: cp.Parameter           # current weight vector w ∈ R^q
    _n_vc_built: int           # vertex count at last problem build
    x: cp.Variable             # decision variable x ∈ R^n
    y: cp.Variable             # image variable y ∈ R^q
    z0: Optional[cp.Variable]  # projection variable (only when m > 0)
    _constraints: List[cp.Constraint]
    _z_vars: List[cp.Variable]  # auxiliary z_i for each vertex (only when m > 0)
    _y_params: List[cp.Parameter]  # y_param for each vertex constraint
    _objective: cp.Maximize
    _problem: cp.Problem

    def __init__(self, graph: Graph, approximation: Approximation, weight: np.ndarray) -> None:
        super().__init__(graph)
        self._approximation = approximation
        self._w = cp.Parameter(graph.q, name="CP_w")
        self._w.value = weight
        self._n_vc_built: int = -1 # no vertex constraints built yet

        self.x = cp.Variable(graph.n, name="CP_x")
        self.y = cp.Variable(graph.q, name="CP_y")
        self.z0 = cp.Variable(graph.m, name="CP_z0") if graph.m > 0 else None

        self._build_problem()

    def _build_problem(self) -> None:
        """Build (or rebuild) the cvxpy problem including all current vertex constraints.
        If a new vertex constraint is added, the problem has to be rebuilt as cvx only allows for static problems."""
        graph = self.graph
        self._constraints = []

        # (x, y) ∈ graph F
        self._constraints += graph.make_constraints(self.x, self.y, self.z0 if graph.m>0 else None)

        # vertex feasibility constraints: for every ȳ_i ∈ vert(approximation) introduce own projection variables z_i
        self._z_vars = []
        self._y_params = []
        if self._approximation.vertices is not None:
            for y_bar in self._approximation.vertices:
                self._add_vertex_constraints(y_bar)

        self._n_vc_built = len(self._approximation.vertices) if self._approximation.vertices is not None else 0

        self._objective = cp.Maximize(self._w @ self.y)
        self._problem = cp.Problem(self._objective, self._constraints)

    def _add_vertex_constraints(self, y_i: np.ndarray) -> None:
        """Add constraints for one vertex ȳ_i: (x, ȳ_i) ∈ graph F, with own z-variable if graph F is a shadow."""
        # wrap y_bar as Parameter so that lambda constraints produce cvxpy constraints
        y_param = cp.Parameter(self.graph.q)
        y_param.value = y_i
        self._y_params.append(y_param)
        if self.graph.m > 0:
            z_i = cp.Variable(self.graph.m, name=f"CP_z_{len(self._z_vars)}")
            self._z_vars.append(z_i)
        else:
            z_i = None
        self._constraints += self.graph.make_constraints(self.x, y_param, z_i if self.graph.m>0 else None)

    def solve(
        self,
        w: np.ndarray,
        *,
        x0: Optional[np.ndarray] = None,
        y0: Optional[np.ndarray] = None,
        solver: Optional[str] = None,
        warm_start: bool = False,
        **solver_kwargs,
    ) -> dict:
        """Solve the convex subproblem for a given weight vector.

        Parameters
        ----------
        w : np.ndarray, shape (q,)
            Weight vector; the objective is ``max w^T y``.
        x0 : np.ndarray, shape (n,), optional
            Warm-start value for the decision variable *x*.
        y0 : np.ndarray, shape (q,), optional
            Warm-start value for the image variable *y*.
        solver : str, optional
            cvxpy solver name.  Falls back to cvxpy default when *None*.
        warm_start : bool
            Passed to cvxpy solve(); reuses the previous solution as a
            starting point.  Default is False.
        **solver_kwargs
            Additional keyword arguments forwarded to the cvxpy solver.

        Returns
        -------
        dict with keys:
            ``"status"``  – cvxpy problem status string.
            ``"optval"``  – optimal objective value (float).
            ``"x"``       – optimal x ∈ R^n (np.ndarray or None).
            ``"y"``       – optimal y ∈ R^q (np.ndarray or None).
        """
        # rebuild if vertices have grown since last build
        current_n = len(self._approximation.vertices) if self._approximation.vertices is not None else 0
        if current_n != self._n_vc_built:
            self._build_problem()

        self._w.value = w  

        # warm-start: feasible starting point from previous iteration
        # note: warm_start needs to be set to True
        if x0 is not None:
            self.x.value = x0
        if y0 is not None:
            self.y.value = y0

        solvers_to_try = ([solver] if solver else []) + ["CLARABEL", "SCS"]
        last_exc: Exception = RuntimeError("No solver succeeded.")
        solved = False
        last_status: Optional[str] = None
        accepted_statuses = {"optimal", "optimal_inaccurate"}
        for _solver in solvers_to_try:
            try:
                self._problem.solve(solver=_solver, warm_start=warm_start, **solver_kwargs)
                last_status = self._problem.status
                if self._problem.status in accepted_statuses:
                    solved = True
                    break
            except Exception as exc:
                last_exc = exc
                continue
        self._solvecount += 1
        if not solved:
            if last_status is not None:
                raise RuntimeError(
                    "CP subproblem could not be solved to optimality. "
                    f"Last solver status: {last_status}."
                )
            raise last_exc

        if self._problem.status == "unbounded":
            raise RuntimeError(
                "Problem CP is unbounded. Possible issue: The algorithm requires graph F to be bounded (compact). "
                "Please add upper/lower bounds to your constraints."
            )

        self.solution = {
            "status": self._problem.status,
            "optval": self._problem.value,
            "x": self.x.value,
            "y": self.y.value,
        }
        return self.solution

    @property
    def solve_count(self) -> int:
        '''returns number of times IP has been solved. Currently unused.'''
        return self._solvecount

    def reset_solve_count(self) -> None:
        self._solvecount = 0

from __future__ import annotations
from typing import  Optional
import numpy as np
import cvxpy as cp

from .graph import Graph
from ._subproblem import SubProblem


class IP(SubProblem):
    """Subproblem (IP(F,x,p,d))

    For point x ∈ dom F, direction d and point p ∈ R^q, p usually ∈ F(x)
        max   t
        s.t.  (x, y) ∈ graph F
               y = p + td
    """

    # instance attributes
    _x: cp.Parameter           # current x ∈ dom F
    _p: cp.Parameter           # current base point p ∈ R^q
    _d: cp.Parameter           # current direction d ∈ R^q
    _n_vc_built: int
    y: cp.Variable             # image point variable
    t: cp.Variable             # scalar step variable
    z0: Optional[cp.Variable]  # projection variable (only when self.graph.m > 0)
    _objective: cp.Maximize
    _problem: cp.Problem

    def __init__(self, graph: Graph, x: np.ndarray, p: np.ndarray, d: np.ndarray) -> None:
        super().__init__(graph)
        self._x = cp.Parameter(graph.n, name="IP_x")
        self._x.value = x
        self._p = cp.Parameter(graph.q, name="IP_p")
        self._p.value = p
        self._d = cp.Parameter(graph.q, name="IP_d")
        self._d.value = d

        self.y = cp.Variable(graph.q, name="IP_y")
        self.t = cp.Variable(name="IP_t")
        self.z0 = cp.Variable(graph.m, name="IP_z0") if graph.m > 0 else None

        self._build_problem()

    def _build_problem(self) -> None:
        """Build (or rebuild) the cvxpy problem"""
        graph = self.graph
        constraints = []

        # (x, y) ∈ graph F
        constraints += graph.make_constraints(self._x, self.y, self.z0 if graph.m>0 else None)

        # y = p + td
        constraints.append(self.y == self._p + self.t * self._d)

        self._objective = cp.Maximize(self.t)
        self._problem = cp.Problem(self._objective, constraints)

    def solve(
        self,
        x: np.ndarray,
        p: np.ndarray,
        d: np.ndarray,
        *,
        solver: Optional[str] = None,
        warm_start: bool = True,
        **solver_kwargs,
    ) -> dict:
        self._x.value = x
        self._p.value = p
        self._d.value = d


        # warm-start: y = p is feasible for t=0
        self.y.value = p
        self.t.value = 0.0

        self._problem.solve(solver=solver, warm_start=warm_start, **solver_kwargs)
        self._solvecount += 1

        if self._problem.status == "unbounded":
            raise RuntimeError(
                "Problem IP is unbounded. Possible issue: The algorithm requires graph F to be bounded (compact). "
                "Please add upper/lower bounds to your constraints."
            )
        if self._problem.status in ("infeasible", "infeasible_inaccurate"):
            raise RuntimeError(
                f"Problem IP is infeasible (status: {self._problem.status}). "
                "The base point p may not lie in F(x), or the graph constraints are contradictory."
            )

        self.solution = {
            "status": self._problem.status,
            "optval": self._problem.value,
            "t": self.t.value,
            "y": self.y.value,
        }
        return self.solution

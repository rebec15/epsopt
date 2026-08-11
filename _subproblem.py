from __future__ import annotations
from abc import ABC, abstractmethod

from .graph import Graph


class SubProblem(ABC):
    """Abstract base class for cvxpy-based auxiliary subproblems.

    Subclasses must implement :meth:`solve`, which builds or updates a
    cvxpy :class:`~cvxpy.Problem`, solves it, and returns a result dict.
    The base class tracks how often :meth:`solve` has been called via
    :attr:`solve_count`.

    Parameters
    ----------
    graph : Graph
        The graph object that provides constraint-building functions.
        :meth:`Graph.validate` is called on construction.
    """

    # instance attributes
    graph: Graph
    _solvecount: int
    solution: dict

    def __init__(self, graph: Graph) -> None:
        graph.validate()
        self.graph = graph
        self._solvecount: int = 0
        self.solution: dict = {}

    @abstractmethod
    def solve(self, *args, **kwargs) -> dict:
        """solves auxiliary problem and gives back solution as dict"""
        ...

    @property
    def solve_count(self) -> int:
        """returns number of solvings"""
        return self._solvecount

    def reset_solve_count(self) -> None:
        """resets solve count"""
        self._solvecount = 0

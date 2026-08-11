# epsopt: an implementation for finding an epsilon-optimizer for compact convex set optimization problems

__version__ = "0.1.0"

from typing import TYPE_CHECKING, Any

from .graph import Graph
from .approximation import Approximation
from .convex_set import ConvexSet
from .cp import CP
from .ip import IP
from .csop import CSOP
from ._subproblem import SubProblem

if TYPE_CHECKING:
    from .convex_set_approximator import ConvexSetApproximator

__all__ = [
    "ConvexSet",
    "Graph",
    "Approximation",
    "SubProblem",
    "CP",
    "IP",
    "CSOP",
    "ConvexSetApproximator",
]


def __getattr__(name: str) -> Any:
    if name == "ConvexSetApproximator":
        from .convex_set_approximator import ConvexSetApproximator

        return ConvexSetApproximator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

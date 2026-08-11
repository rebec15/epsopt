"""
exp_plus_rplus_csop_example.py

CSOP example for
    F(x) = {exp(-x)} + R_+
on a bounded domain x in [0, 2].

Graph model:
    gr F = { (x, y) in R x R :
             0 <= x,
             y >= exp(-x) }

Recession cone in value space y:
    K = R_+ = cone({1}).
"""

from __future__ import annotations

import os
import sys

import cvxpy as cp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from epsopt import CSOP, Graph


def main() -> None:
    # q=1 value space, nontrivial recession cone K = R_+ with generator +1.
    graph = Graph(
        n=1,
        q=1,
        m=0,
        name="ExpPlusRPlus",
        recession_cone_generators=[[1.0]],
    )

    graph.add_constraint_fn(
        lambda x, y, z: [
            x[0] >= 0.0,
            y[0] >= cp.exp(-x[0]),
        ]
    )

    # Choose y0 in img F. Since img F = [0, +inf), y0=0.4 is feasible.
    y0 = np.array([0.4], dtype=float)
    eps = 1e-2

    csop = CSOP(graph, solver="SCS")
    result = csop.computeEpsOptimizer(
        y=y0,
        eps=eps,
        compute_gamma_upper_bound=True,
    )

    x_star = float(np.asarray(result["x"], dtype=float).ravel()[0])
    y_floor = float(np.exp(-x_star))
    y0_ok = bool(y0[0] >= y_floor - 1e-7)

    print("=== CSOP Example: F(x) = {exp(-x)} + R_+ ===")
    print(f"y0                 = {y0}")
    print(f"eps                = {eps}")
    print(f"x*                 = {result['x']}")
    print(f"exp(-x*)           = {y_floor:.8f}")
    print(f"y0 in F(x*)        = {y0_ok}")
    print(f"gamma_upper_bound  = {result.get('gamma_upper_bound')}")
    print(f"approximation      = {result['approx']}")


if __name__ == "__main__":
    main()

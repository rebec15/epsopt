"""
simple_test.py – minimal example to test epsopt-package. Compact graph, recc = {0}.

Problem (uni-criterial, n=1, q=1):
    x ∈ [0, 1]
    F(x) = { y ∈ R : x ≤ y[0] ≤ x+1 }
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",".."))

import numpy as np
from epsopt_v2 import Graph, CSOP

# Define graph
# n=1: x ∈ R, q=2: y ∈ R^2, no shadow (m=0)
graph = Graph(n=1, q=1, name="SimpleTest")

graph.add_constraint_fn(lambda x, y, z: [
    y[0] >= x[0],        # y[0] >= x
    y[0] <= x[0] + 1,   # y[0] <= x+1  (upper bound)
    x[0] >= 0,
    x[0] <= 1,
])

# Define csop and test algorithm
csop = CSOP(graph)

y0 = np.array([0.5])
eps = 0.1

result = csop.computeEpsOptimizer(y0, eps)

print("=== Result ===")
print(f"optimizer x*    = {result['x']}")
print(f"chosen y   = {result['y_choice']}")
print(f"eps   = {result['eps']}")
print(f"Approximation: {result['approx']}")

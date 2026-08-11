"""
convex_set_approximator_test.py - minimal test for ConvexSetApproximator.

Set definition (n=1, q=1, m=0):
    S = { (x, y) in R^2 : 0 <= x <= 1, x <= y <= x + 1 }

The approximator lifts S to a graph of a constant map F with dom F = {0},
then runs the epsilon-optimizer algorithm on that graph to return an inner approximation of S.

Additional test:
    DiskSet = { x in R^2 : ||x||_2 <= 1 }

3D test:
    S3D = { (x1, x2, y) in R^3 : 0 <= x1 <= 1, 0 <= x2 <= 1, x1+x2 <= y <= x1+x2+0.5 }

Unbounded test with positive orthant recession cone:
    SPlus = { (x, y) in R^2 : x >= 0, y >= 0 }
    recc(SPlus) = R_+^2

Unbounded 3D test with positive orthant recession cone:
    SPlus3D = { (x1, x2, y) in R^3 : x1 >= 0, x2 >= 0, y >= 0 }
    recc(SPlus3D) = R_+^3
"""

from __future__ import annotations

import os
import sys

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from epsopt_v2 import ConvexSet, ConvexSetApproximator


def build_set() -> ConvexSet:
    set_s = ConvexSet(n=1, q=1, m=0, name="SimpleSet")
    set_s.add_constraint_fn(
        lambda x, y, z: [
            x[0] >= 0,
            x[0] <= 1,
            y[0] >= x[0],
            y[0] <= x[0] + 1,
        ]
    )
    return set_s


def build_disk_set() -> ConvexSet:
    set_s = ConvexSet(n=2, q=0, m=0, name="DiskSet")
    set_s.add_constraint_fn(
        lambda x, y, z: [
            cp.norm(x, 2) <= 1,
        ]
    )
    return set_s


def build_3d_set() -> ConvexSet:
    set_s = ConvexSet(n=2, q=1, m=0, name="S3D")
    set_s.add_constraint_fn(
        lambda x, y, z: [
            x[0] >= 0,
            x[0] <= 1,
            x[1] >= 0,
            x[1] <= 1,
            y[0] >= x[0] + x[1],
            y[0] <= x[0] + x[1] + 0.5,
        ]
    )
    return set_s


def build_unbounded_orthant_set() -> ConvexSet:
    set_s = ConvexSet(
        n=1,
        q=1,
        m=0,
        name="SPlus",
        recession_cone_generators=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )
    set_s.add_constraint_fn(
        lambda x, y, z: [
            x[0] >= 0,
            y[0] >= 0,
        ]
    )
    return set_s


def build_unbounded_orthant_3d_set() -> ConvexSet:
    set_s = ConvexSet(
        n=2,
        q=1,
        m=0,
        name="SPlus3D",
        recession_cone_generators=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    set_s.add_constraint_fn(
        lambda x, y, z: [
            x[0] >= 0,
            x[1] >= 0,
            y[0] >= 0,
        ]
    )
    return set_s


def main() -> None:
    eps = 0.1
    tests = np.array([False, True, False, False, False], dtype=bool)  # choose test sets

    if(tests[0]):
        # Test 1: S in R^(n+q) with n=1, q=1.
        set_interval = build_set()
        approximator_interval = ConvexSetApproximator(set_interval)
        y0_interval = np.array([0.2, 0.6], dtype=float)
        result_interval = approximator_interval.approximate(y=y0_interval, eps=eps)

        print("=== ConvexSetApproximator Test 1 (SimpleSet) ===")
        print(f"set name         = {set_interval.name}")
        print(f"reference y0     = {result_interval['y_choice']}")
        print(f"eps              = {result_interval['eps']}")
        print(f"optimizer x*     = {result_interval['x']}")
        print(f"approximation    = {result_interval['approx']}")
        approximator_interval.plot_approximation(
            result_interval["approx"],
            y_choice=result_interval["y_choice"],
            eps=eps,
            set_description="{(x,y) in R^2 : 0 <= x <= 1, x <= y <= x + 1}",
            approx_vertex_size=0.0,
        )

    if(tests[1]):
        # Test 2: S as unit disk in R^2 (q=0 case).
        set_disk = build_disk_set()
        approximator_disk = ConvexSetApproximator(set_disk)
        y0_disk = np.array([0.2, 0.3], dtype=float)
        result_disk = approximator_disk.approximate(y=y0_disk, eps=eps)

        print("=== ConvexSetApproximator Test 2 (DiskSet) ===")
        print(f"set name         = {set_disk.name}")
        print(f"reference y0     = {result_disk['y_choice']}")
        print(f"eps              = {result_disk['eps']}")
        print(f"optimizer x*     = {result_disk['x']}")
        print(f"approximation    = {result_disk['approx']}")
        approximator_disk.plot_approximation(
            result_disk["approx"],
            y_choice=result_disk["y_choice"],
            eps=eps,
            set_description="{x in R^2 : ||x||_2 <= 1}",
            approx_vertex_size=0.0,
        )

    if(tests[2]):
        # Test 3: 3D set with n+q=3 to test 3D plotting branch.
        set_3d = build_3d_set()
        approximator_3d = ConvexSetApproximator(set_3d)
        y0_3d = np.array([0.2, 0.3, 0.7], dtype=float)
        result_3d = approximator_3d.approximate(y=y0_3d, eps=eps)

        print("=== ConvexSetApproximator Test 3 (S3D) ===")
        print(f"set name         = {set_3d.name}")
        print(f"reference y0     = {result_3d['y_choice']}")
        print(f"eps              = {result_3d['eps']}")
        print(f"optimizer x*     = {result_3d['x']}")
        print(f"approximation    = {result_3d['approx']}")
        approximator_3d.plot_approximation(
            result_3d["approx"],
            y_choice=result_3d["y_choice"],
            eps=eps,
            set_description="{(x1,x2,y) in R^3 : 0 <= x1 <= 1, 0 <= x2 <= 1, x1+x2 <= y <= x1+x2+0.5}",
            approx_vertex_size=0.0,
        )

    if(tests[3]):
        # Test 4: Unbounded set in R^2 with recc(S) = R_+^2.
        set_plus = build_unbounded_orthant_set()
        approximator_plus = ConvexSetApproximator(set_plus)
        y0_plus = np.array([0.3, 0.4], dtype=float)
        result_plus = approximator_plus.approximate(y=y0_plus, eps=eps)

        print("=== ConvexSetApproximator Test 4 (SPlus, unbounded) ===")
        print(f"set name         = {set_plus.name}")
        print(f"reference y0     = {result_plus['y_choice']}")
        print(f"eps              = {result_plus['eps']}")
        print(f"optimizer x*     = {result_plus['x']}")
        print(f"approximation    = {result_plus['approx']}")
        print(f"recc generators  = {set_plus.recession_cone_generators}")
        approximator_plus.plot_approximation(
            result_plus["approx"],
            y_choice=result_plus["y_choice"],
            eps=eps,
            set_description="{(x,y) in R^2 : x >= 0, y >= 0} with recc(S)=R_+^2",
            approx_vertex_size=0.0,
        )

    if(tests[4]):
        # Test 5: Unbounded 3D set in R^3 with recc(S) = R_+^3.
        set_plus_3d = build_unbounded_orthant_3d_set()
        approximator_plus_3d = ConvexSetApproximator(set_plus_3d)
        y0_plus_3d = np.array([0.2, 0.3, 0.4], dtype=float)
        result_plus_3d = approximator_plus_3d.approximate(y=y0_plus_3d, eps=eps)

        print("=== ConvexSetApproximator Test 5 (SPlus3D, unbounded) ===")
        print(f"set name         = {set_plus_3d.name}")
        print(f"reference y0     = {result_plus_3d['y_choice']}")
        print(f"eps              = {result_plus_3d['eps']}")
        print(f"optimizer x*     = {result_plus_3d['x']}")
        print(f"approximation    = {result_plus_3d['approx']}")
        print(f"recc generators  = {set_plus_3d.recession_cone_generators}")
        approximator_plus_3d.plot_approximation(
            result_plus_3d["approx"],
            y_choice=result_plus_3d["y_choice"],
            eps=eps,
            set_description="{(x1,x2,y) in R^3 : x1 >= 0, x2 >= 0, y >= 0} with recc(S)=R_+^3",
            approx_vertex_size=0.0,
        )

    plt.show()


if __name__ == "__main__":
    main()

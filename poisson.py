import sys

import numpy as np
import scipy.sparse.linalg

from conditions import Neumann
from helpers import central_difference
from integrate import integrate
from interpolate import calculate_poisson_derivatives, interpolate
from nonuniform import has_uniform_steps, liu_coefficients
from refine import refine_after, select_max

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsmr.html
explain_istop = {
    0: "x=0 is a solution.  If x0 was given, then x=x0 is a solution.",
    1: "x is an approximate solution to A*x = B, according to atol and btol.",
    2: "x approximately solves the least-squares problem according to atol.",
    3: "COND(A) seems to be greater than CONLIM.",
    4: (
        "x is an approximate solution to A*x = B, according to atol and btol with"
        "atol = btol = eps (machine precision)"
    ),
    5: (
        "x approximately solves the least-squares problem according to atol with"
        "atol = eps."
    ),
    6: "COND(A) seems to be greater than CONLIM with CONLIM = 1/eps.",
    7: "ITN reached maxiter before the other stopping conditions were satisfied.",
}

bad_istops = (3, 6, 7)


def poisson(f, M, conditions, maxiter=1e6, explain_solution=True):
    assert M >= 1
    assert len(conditions) == 2

    h = 1 / (M + 1)
    A = central_difference(M + 2, power=2) / h ** 2

    # x1=h, x2=2h, ..., xm+1 = 1
    x = np.arange(0, M + 2).astype(np.float64) * h
    f = f(x)

    for condition in conditions:
        vector = condition.get_vector(length=M + 2, h=h)
        A[condition.m, :] = vector
        f[condition.m] = condition.get_scalar()

    sparse = scipy.sparse.csc_matrix(A)

    if isinstance(conditions[0], Neumann) and isinstance(conditions[1], Neumann):
        # Ill posed equation -> use a least squares solution instead
        # The norms of A and f scale rapidly with M, so we set the tolerance to
        # machine precision, and limit ourselves to `iter_lim` iterations
        solution = scipy.sparse.linalg.lsmr(
            sparse, f, maxiter=maxiter, atol=0, btol=0, conlim=0
        )
        U = solution[0]
        istop = solution[1]
        if istop in bad_istops:
            print(
                "Error when solving least squares problem:",
                explain_istop[istop],
                file=sys.stderr,
            )
        elif explain_solution:
            print(
                "Solved least squares problem:", explain_istop[istop], file=sys.stderr
            )
    else:
        U = scipy.sparse.linalg.spsolve(sparse, f)

    return x, U


def poisson_4_point(f, x, conditions):
    """
    Solve Poisson's equation on the mesh x

    Uses the method for arbitrary mesh sizes described in:
    Liu Jianchun, Gary A. Pope, Kamy Sepehrnoori,
    A high-resolution finite-difference scheme for nonuniform grids,
    Applied Mathematical Modelling,
    Volume 19, Issue 3,
    1995,
    Pages 162-172,
    ISSN 0307-904X,
    https://doi.org/10.1016/0307-904X(94)00020-7.
    """
    assert has_uniform_steps(x, (0, 1)), "First two spaces must be of equal length"
    assert len(conditions) == 2
    assert not isinstance(conditions[0], Neumann) or not isinstance(
        conditions[1], Neumann
    ), "The problem is ill posed"

    length = x.shape[0]
    A = scipy.sparse.lil_matrix((length, length), dtype=np.float64)

    # Use the three point central difference as the first condition
    A[1, 0:3] = (1, -2, 1) / (x[1] - x[0]) ** 2

    # Populate the matrix with the four point formula
    for i in range(2, length - 1):
        a, b, c = liu_coefficients(x, i, order=2)
        A[i, i - 2 : i + 2] = (a, b, -(a + b + c), c)

    f = f(x)

    steps = x[1:] - x[:-1]
    for condition in conditions:
        vector = condition.get_vector(length=length, h=steps[condition.m])
        context = np.nonzero(vector)
        indicies = np.arange(np.min(context), np.max(context))
        assert has_uniform_steps(
            x, indicies
        ), "Step sizes for boundary conditions must be uniform"

        A[condition.m, :] = vector
        f[condition.m] = condition.get_scalar()

    sparse = scipy.sparse.csc_matrix(A)

    U = scipy.sparse.linalg.spsolve(sparse, f)

    return x, U


def amr(f, u, conditions, amt_points_target, select_refinement=select_max):
    x = np.array((0, 0.5, 1), dtype=np.float64)
    to_refine = np.array((), dtype=np.int32)

    while x.shape[0] < amt_points_target:
        x = refine_after(x, to_refine)
        x, U = poisson_4_point(f, x, conditions)
        interpolated = interpolate(x, U, calculate_poisson_derivatives(f))
        err = integrate(lambda x: (u(x) - interpolated(x)) ** 2, x[:-1], x[1:])
        to_refine = select_refinement(err)

    return x, U

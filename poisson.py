import sys

import numpy as np
import scipy.sparse.linalg

from conditions import Neumann
from helpers import central_difference

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


def poisson(f, M, condition_1, condition_2, maxiter=1e6, explain_solution=True):
    assert M >= 4

    h = 1 / (M + 1)
    A = central_difference(M + 2, power=2) / h ** 2

    # x1=h, x2=2h, ..., xm+1 = 1
    x = np.arange(0, M + 2).astype(np.float64) * h
    f = f(x)

    A[0, :] = condition_1.get_vector(length=M + 2, h=h)
    f[0] = condition_1.get_scalar()

    A[-1, :] = condition_2.get_vector(length=M + 2, h=h)
    f[-1] = condition_2.get_scalar()

    sparse = scipy.sparse.csc_matrix(A)

    if isinstance(condition_1, Neumann) and isinstance(condition_2, Neumann):
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

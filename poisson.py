import numpy as np
import scipy.sparse.linalg

from helpers import central_difference


def poisson(f, M, condition_1, condition_2):
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

    U = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), f)

    return x, U

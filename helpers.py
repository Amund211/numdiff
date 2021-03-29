import numpy as np
import scipy.sparse


def central_difference(N, power=2):
    if power == 1:
        return scipy.sparse.diags(
            (-1, 0, 1), (-1, 0, 1), shape=(N, N), format="lil", dtype=np.float64
        )
    elif power == 2:
        return scipy.sparse.diags(
            (1, -2, 1), (-1, 0, 1), shape=(N, N), format="lil", dtype=np.float64
        )


def l2(v):
    """Calculate the discrete l2 norm of the vector v"""
    l = v.shape[0]
    return np.sqrt(np.sum(np.square(v)) / l)


def relative_l2_error(u, U):
    """Calculate the relative discrete l2 norm of u - U"""
    return l2(u - U) / l2(u)

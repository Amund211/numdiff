import numpy as np
import scipy.sparse

from integrate import composite


def central_difference(N, power=2, format="lil"):
    if power == 1:
        return scipy.sparse.diags(
            (-1 / 2, 0, 1 / 2),
            (-1, 0, 1),
            shape=(N, N),
            format=format,
            dtype=np.float64,
        )
    elif power == 2:
        return scipy.sparse.diags(
            (1, -2, 1), (-1, 0, 1), shape=(N, N), format=format, dtype=np.float64
        )


def l2(v, axis=None):
    """Calculate the discrete l2 norm of the vector v"""
    l = v.shape[0]
    return np.sqrt(np.sum(np.square(v), axis=axis) / l)


def relative_l2_error(u, U):
    """Calculate the relative discrete l2 norm of u - U"""
    return l2(u - U) / l2(u)


def L2(f, x):
    """Calculate the continuous L2 norm of the function f on the grid x"""
    return np.sqrt(composite(lambda x: f(x) ** 2, x))


def relative_L2_error(u, U, x):
    """Calculate the relative continuous L2 norm of u - U on the grid x"""
    return L2(lambda x: u(x) - U(x), x) / L2(u, x)

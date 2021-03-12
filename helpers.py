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


def discrete_l2(v):
    l = v.shape[0]
    return np.sqrt(np.sum(np.square(v)) / l)


def relative_discrete_l2(u, U):
    return discrete_l2(u - U) / discrete_l2(u)

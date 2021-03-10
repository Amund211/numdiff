import numpy as np


def embed(array, length):
    embedded = np.zeros(length)
    embedded[: array.shape[0]] = array
    return embedded


def central_difference(N, power=2):
    if power == 1:
        ones = np.ones(N - 1)
        return (np.diag(-ones, k=-1) + np.diag(ones, k=1)).astype(np.float64)
    elif power == 2:
        diag = -2 * np.ones(N)
        offdiag = np.ones(N - 1)

        return (np.diag(diag) + np.diag(offdiag, k=-1) + np.diag(offdiag, k=1)).astype(
            np.float64
        )


def discrete_l2(v):
    l = v.shape[0]
    return np.sqrt(np.sum(np.square(v)) / l)


def relative_discrete_l2(u, U):
    return discrete_l2(u - U) / discrete_l2(u)

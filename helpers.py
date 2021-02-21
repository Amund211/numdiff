import numpy as np


def discrete_l2(v):
    l = v.shape[0]
    return np.sqrt(np.sum(np.square(v)) / l)


def relative_discrete_l2(u, U):
    return discrete_l2(u - U) / discrete_l2(u)

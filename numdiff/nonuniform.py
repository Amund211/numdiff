"""
Utility functions for nonuniform grids
"""

import numpy as np


def has_uniform_steps(x, indicies):
    """Check that each of the indicies i of the step-length vector are equal"""
    indicies = np.array(indicies, dtype=np.int32)
    if indicies.shape[0] <= 1:
        return True

    steps = x[1:] - x[:-1]
    return np.allclose(steps[indicies], steps[indicies[0]])


def liu_coefficients(x, i, order):
    """
    Return the coefficients for the finite difference operator on a nonuniform grid

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
    # di[n] = d_{i + n} = the distance from element i to element i + n
    di = (0, x[i + 1] - x[i], x[i] - x[i - 2], x[i] - x[i - 1])

    if order == 1:
        a = di[-1] * di[1] / (di[-2] * (di[-2] + di[1]) * (di[-2] - di[-1]))
        b = -di[-2] * di[1] / (di[-1] * (di[-2] - di[-1]) * (di[-1] + di[1]))
        c = di[-2] * di[-1] / (di[1] * (di[-1] + di[1]) * (di[-2] + di[1]))
    elif order == 2:
        a = 2 * (di[1] - di[-1]) / (di[-2] * (di[-2] + di[1]) * (di[-2] - di[-1]))
        b = 2 * (di[-2] - di[1]) / (di[-1] * (di[-2] - di[-1]) * (di[-1] + di[1]))
        c = 2 * (di[-2] + di[-1]) / (di[1] * (di[-1] + di[1]) * (di[-2] + di[1]))
    else:
        raise NotImplementedError("That order is not implemented")

    return a, b, c

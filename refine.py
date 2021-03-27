import numpy as np

from helpers import relative_discrete_l2


def refine_after(x, indicies):
    """Halve the step size between i and i+1 for each i in indicies"""
    return np.insert(x, indicies + 1, x[indicies] + (x[indicies + 1] - x[indicies]) / 2)


def refine_symmetric(x, indicies):
    """
    Refine to the left and right of each i

    Makes sure that the two first steps are of equal length
    """
    indicies = np.unique(np.concatenate((indicies - 1, indicies)))
    indicies = indicies[np.logical_and(indicies >= 0, indicies < x.shape[0] - 1)]
    if 1 in indicies and 0 not in indicies:
        indicies = np.insert(indicies, 0, 0)

    return refine_after(x, indicies)


def refine_mesh(solver, M_range, analytical, calculate_distance):
    """Solve a numerical scheme for a range of M-values and return the error for each"""
    distances = np.empty(M_range.shape, dtype=np.float64)
    amt_points = np.empty(M_range.shape, dtype=np.int32)
    for i, M in enumerate(M_range):
        x, numerical = solver(M)
        distances[i] = calculate_distance(x, analytical, numerical)
        amt_points[i] = x.shape[0]

    return amt_points, distances


def calculate_relative_discrete_l2(x, analytical, numerical):
    """
    Helper to calculate e^r_l2

    The analytical solution is a function and the numerical solution are the
    values at the grid points.
    """
    return relative_discrete_l2(analytical(x), numerical)


def make_solver(cls, f, **kwargs):
    """
    Create a solver function from a time evolution scheme for use in `refine_mesh`
    """

    def solver(M):
        scheme = cls(M=M, **kwargs)
        x_axis, solution = scheme.solve(f)
        return x_axis, solution[:, -1]

    return solver
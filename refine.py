import numpy as np

from helpers import relative_l2_error, relative_L2_error
from interpolate import interpolate


def refine_after(x, indicies):
    """Halve the step size between i and i+1 for each i in indicies"""
    return np.insert(x, indicies + 1, (x[indicies + 1] + x[indicies]) / 2)


def ensure_uniform_steps(indicies, amt_intervals):
    """
    Ensure that the two first and last steps have uniform step lengths after refinement
    """
    if 1 in indicies and 0 not in indicies:
        indicies = np.insert(indicies, 0, 0)
    if amt_intervals - 2 in indicies and amt_intervals - 1 not in indicies:
        indicies = np.insert(indicies, 0, amt_intervals - 1)

    return indicies


def select_max(err, alpha=0.7):
    """Select indicies to refine where the error exceeds `alpha` * max(err)"""
    max_err = np.max(err)
    to_refine = np.flatnonzero(err > 0.7 * max_err)

    return ensure_uniform_steps(to_refine, err.shape[0])


def refine_mesh(solver, M_range, analytical, calculate_distance):
    """
    Solve a numerical scheme for a range of M-values and return the error for each

    `calculate_distance` should take the grid, the analytical function and the
    value returned by the solver (function or values at gridpoints)
    """
    distances = np.empty(M_range.shape, dtype=np.float64)
    amt_points = np.empty(M_range.shape, dtype=np.int32)
    for i, M in enumerate(M_range):
        x, numerical = solver(M)
        distances[i] = calculate_distance(x, analytical, numerical)
        amt_points[i] = x.shape[0]

    return amt_points, distances


def calculate_relative_l2_error(x, analytical, numerical):
    """Helper to calculate discrete e^r_l2"""
    return relative_l2_error(analytical(x), numerical)


def calculate_relative_L2_error(x, analytical, numerical):
    """Helper to calculate continuous e^r_l2"""
    return relative_L2_error(analytical, numerical, x)


def make_solver(cls, f, interpolate_result=False, bc_type="not-a-knot", **kwargs):
    """
    Create a solver function from a time evolution scheme for use in `refine_mesh`
    """

    def solver(M):
        scheme = cls(M=M, **kwargs)
        x_axis, solution = scheme.solve(f)
        if interpolate_result:
            return x_axis, interpolate(x_axis, solution[:, -1], bc_type=bc_type)
        else:
            return x_axis, solution[:, -1]

    return solver

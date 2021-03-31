import numpy as np


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
    to_refine = np.flatnonzero(err >= alpha * max_err)

    return ensure_uniform_steps(to_refine, err.shape[0])


def select_avg(err, alpha=1.0):
    """Select indicies to refine where the error exceeds `alpha` * avg(err)"""
    avg_err = np.average(err)
    to_refine = np.flatnonzero(err >= alpha * avg_err)

    return ensure_uniform_steps(to_refine, err.shape[0])


def refine_mesh(solver, param_range, analytical, calculate_distance):
    """
    Solve a numerical scheme for a range of parameters and return the error for each

    `calculate_distance` should take the grid, the analytical function and the
    value returned by the solver (function or values at gridpoints)
    """
    distances = np.empty(param_range.shape, dtype=np.float64)
    ndofs = np.empty(param_range.shape, dtype=np.int32)
    for i, param in enumerate(param_range):
        x, numerical, ndof = solver(param)
        distances[i] = calculate_distance(x, analytical, numerical)
        ndofs[i] = ndof

    return ndofs, distances

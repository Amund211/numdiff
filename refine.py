import numpy as np


def refine_after(x, indicies):
    """Halve the step size between i and i+1 for each i in indicies"""
    return np.insert(x, indicies + 1, (x[indicies + 1] + x[indicies]) / 2)


def ensure_uniform_steps(indicies, err, max_refinement):
    """
    Ensure that the two first and last steps have uniform step lengths after refinement
    """
    # Find the missing indicies
    missing = np.array((), dtype=np.int32)
    amt_intervals = err.shape[0]
    if 1 in indicies and 0 not in indicies:
        missing = np.append(missing, 0)
    if amt_intervals - 2 in indicies and amt_intervals - 1 not in indicies:
        missing = np.append(missing, amt_intervals - 1)

    # Make sure that max_refinement indicies isn't exceeded
    if indicies.shape[0] + missing.shape[0] > max_refinement:
        amount_to_refine = max_refinement - missing.shape[0]
        largest_indicies = np.argpartition(err[indicies], -amount_to_refine)[
            -amount_to_refine:
        ]
        indicies = indicies[largest_indicies]

    return np.append(indicies, missing)


def select_max(err, max_refinement, alpha=0.7):
    """Select indicies to refine where the error exceeds `alpha` * max(err)"""
    max_err = np.max(err)
    to_refine = np.flatnonzero(err >= alpha * max_err)

    return ensure_uniform_steps(to_refine, err, max_refinement)


def select_avg(err, max_refinement, alpha=1.0):
    """Select indicies to refine where the error exceeds `alpha` * avg(err)"""
    avg_err = np.average(err)
    to_refine = np.flatnonzero(err >= alpha * avg_err)

    return ensure_uniform_steps(to_refine, err, max_refinement)


def refine_mesh(solver, param_range, analytical, calculate_distances):
    """
    Solve a numerical scheme for a range of parameters and return the error for each

    Each function in `calculate_distances` should take the grid, the analytical
    function, and the values at gridpoints
    """
    distances_list = [None] * len(calculate_distances)
    for i in range(len(distances_list)):
        distances_list[i] = np.empty(param_range.shape, dtype=np.float64)

    ndofs = np.empty(param_range.shape, dtype=np.int32)
    for i, param in enumerate(param_range):
        x, numerical, ndof = solver(param)
        for distances, calculate_distance in zip(distances_list, calculate_distances):
            distances[i] = calculate_distance(x, analytical, numerical)
        ndofs[i] = ndof

    return ndofs, distances_list

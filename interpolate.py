import numpy as np
from scipy.interpolate import BPoly

from nonuniform import has_uniform_steps, liu_coefficients


def interpolate(x, y, calculate_derivatives):
    """
    Return a spline interpolation of y, with the given derivatives

    `calculate_derivatives` should take x and y, and return a list of lists where
    ret[i][j] is the (i+1)-th derivative at x[j]. Each list ret[i] should have the same
    length as x.
    """
    return BPoly.from_derivatives(x, list(zip(y, *calculate_derivatives(x, y))))


def calculate_poisson_derivatives(f):
    def finite_difference_approx(x, y, i):
        """Helper for first derivative"""
        maxi = x.shape[0] - 1

        if i == 0:
            h = x[1] - x[0]
            return (-3 / 2 * y[0] + 2 * y[1] - 1 / 2 * y[2]) / h
        elif i == 1 or i == maxi - 1:
            h = x[i] - x[i - 1]
            return (y[i + 1] - y[i - 1]) / (2 * h)
        elif i == maxi:
            h = x[i] - x[i - 1]
            return (1 / 2 * y[i - 2] - 2 * y[i - 1] + 3 / 2 * y[i]) / h
        else:
            a, b, c = liu_coefficients(x, i, order=1)
            return a * y[i - 2] + b * y[i - 1] - (a + b + c) * y[i] + c * y[i + 1]

    def calculate_derivatives(x, y):
        assert has_uniform_steps(x, (0, 1)), "First two spaces must be of equal length"
        assert has_uniform_steps(x, (-1, -2)), "Last two spaces must be of equal length"

        first_order = [finite_difference_approx(x, y, i) for i in range(len(x))]
        second_order = f(x)
        return [first_order, second_order]

    return calculate_derivatives

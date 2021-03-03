from functools import cache, cached_property

import numpy as np
import scipy.sparse.linalg

from helpers import central_difference


class Equation:
    """
    Specify the discretization of the operator L, L_h, in `.operator()` as a matrix.

    If the operator isn't linear you can override `.get_operator()`
    """

    @cached_property
    def free_indicies(self):
        """
        np.array of indicies that are not calculated by the scheme
        These are free to be used as conditions
        """

        raise NotImplementedError

    @cached_property
    def restricted_x_indicies(self):
        """Get the x axis indicies where this method has all its needed context"""

        unrestricted = np.arange(0, self.M + 2)
        return unrestricted[np.isin(unrestricted, self.free_indicies, invert=True)]

    def operator(self):
        """A matrix representing the discretized operator L_h"""

        raise NotImplementedError

    @cache
    def get_operator(self):
        """
        Return a function that applies the operator to a given vector
        """
        sparse = scipy.sparse.csr_matrix(self.operator())
        return lambda v: sparse @ v

    def apply_operator(self, n, v, restrict=True):
        """Apply the discretized operator in x to the vector v"""

        res = self.get_operator()(v)

        if restrict:
            return res[self.restricted_x_indicies]
        else:
            # Assumes independent conditions
            for condition in self.conditions:
                res[condition.m] = condition.solve_restricted(v, self.M + 2, self.h, t)
            return res

    def restrict(self, v):
        """Restrict a vector to self.restricted_x_indicies"""

        return v[self.restricted_x_indicies]


class HeatEquation(Equation):
    @cached_property
    def free_indicies(self):
        return np.array((0, self.M + 1), dtype=np.int64)

    def operator(self):
        return central_difference(self.M + 2, power=2) / self.h ** 2


class InviscidBurgers(Equation):
    @cached_property
    def free_indicies(self):
        return np.array((0, self.M + 1), dtype=np.int64)

    @cache
    def get_operator(self):
        def operator(v):
            res = np.empty((self.M + 2,), dtype=np.float64)
            res[self.restricted_x_indicies] = -v[1:-1] / (2 * self.h) * (v[2:] - v[:-2])
            return res

        return operator


class InviscidBurgers2(InviscidBurgers):
    @cache
    def get_operator(self):
        def operator(v):
            res = np.empty((self.M + 2,), dtype=np.float64)
            res[self.restricted_x_indicies] = -(v[2:] ** 2 - v[:-2] ** 2) / (4 * self.h)
            return res

        return operator

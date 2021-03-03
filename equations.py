from functools import cache, cached_property

import numpy as np
import scipy.sparse.linalg

from helpers import central_difference


class Equation:
    """
    Specify the discretization of the operator L, L_h, in `.operator()` as a matrix.

    If the operator isn't linear you can override `.apply_operator()`
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
    def get_csr_operator(self):
        return scipy.sparse.csr_matrix(self.operator())

    def apply_operator(self, v, restrict=True):
        """Apply the discretized operator in x to the vector v"""

        res = self.get_csr_operator() @ v

        return res[self.restricted_x_indicies] if restrict else res

    def restrict(self, v):
        """Restrict a vector to self.restricted_x_indicies"""

        return v[self.restricted_x_indicies]


class HeatEquation(Equation):
    @cached_property
    def free_indicies(self):
        return np.array((0, self.M + 1), dtype=np.int64)

    def operator(self):
        return central_difference(self.M + 2, power=2) / self.h ** 2
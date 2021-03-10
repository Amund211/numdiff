from functools import cache, cached_property

import numpy as np
import scipy.sparse.linalg

from helpers import embed


class Equation:
    """
    Specify the discretization of the operator L, L_h, in `.operator()` as a matrix.

    If the operator isn't linear you can override `.get_operator()`
    """

    def __init__(self, *, M, conditions, **kwargs):
        self.M = M
        self.conditions = conditions
        super().__init__(**kwargs)

        assert self.M > 0

    @cached_property
    def h(self):
        return 1 / (self.M + 1)

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

    @cached_property
    def operator_info(self):
        """
        The index of the center of the discretized operator along with the amount of
        needed left and right context

        Defaults based on self.single_operator
        """

        operator = self.single_operator(1)
        length = operator.shape[0]
        if length % 2 == 1:
            center = length // 2
            return center - 0, center, length - 1 - center
        else:
            raise ValueError(
                "Operator was not odd length. Could not find center index."
            )

    @cache
    def single_operator(self, h):
        """np.array describing the discretized operator"""

        raise NotImplementedError

    def get_single_operator_for(self, h, index):
        left_context, center, right_context = self.operator_info
        if index - left_context >= 0 and index + right_context <= self.M + 1:
            return np.roll(
                embed(self.single_operator(self.h), self.M + 2), index - center
            )
        else:
            return np.zeros(self.M + 2)

    @cache
    def operator(self):
        """A matrix representing the discretized operator L_h"""
        return np.array(
            [self.get_single_operator_for(self.h, i) for i in range(self.M + 2)],
            dtype=np.float64,
        )

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
                res[condition.m] = condition.solve_restricted(
                    v, self.M + 2, self.h, n * self.k
                )
            return res

    def restrict(self, v):
        """Restrict a vector to self.restricted_x_indicies"""

        return v[self.restricted_x_indicies]


class HeatEquation(Equation):
    @cached_property
    def free_indicies(self):
        return np.array((0, self.M + 1), dtype=np.int64)

    @cache
    def single_operator(self, h):
        return np.array((1, -2, 1), dtype=np.float64) / self.h ** 2


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


class PeriodicKdV(Equation):
    """Linearized Korteweg-deVries with periodic boundary condition with period 2"""

    @cached_property
    def free_indicies(self):
        return np.array([self.M + 1], dtype=np.int64)

    def single_operator(self, h):
        # Since the equation is on [-1, 1] we introduce a shift in x: 2 * (x-1/2) to
        # solve it on [0, 1] instead. By solving this alternate diff. eqn we introduce a
        # factor 2 for each power of the derivative, so we divide the operator by 2**p.

        d3 = np.array((-1, 0, 3, 0, -3, 0, 1), dtype=np.float64) / (8 * h ** 3) / 2 ** 3
        d1 = np.array((0, 0, -1, 0, 1, 0, 0), dtype=np.float64) / (2 * h) / 2

        return -d3 - (1 + np.pi ** 2) * d1

    def get_single_operator_for(self, h, index):
        left_context, center, right_context = self.operator_info
        # The first and last index are the same value, so we roll on the index before
        return embed(
            np.roll(embed(self.single_operator(self.h), self.M + 1), index - center),
            self.M + 2,
        )

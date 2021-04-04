from functools import cache, cached_property

import numpy as np
import scipy.sparse.linalg

from helpers import central_difference


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
    def x_indicies(self):
        """The x indicies that will be solved for x = i * h"""
        return np.arange(0, self.M + 2)

    @cached_property
    def restricted_indicies(self):
        """np.array of indicies that are restricted by the boundary conditions"""
        raise NotImplementedError

    @cached_property
    def free_indicies(self):
        """np.array of indicies where the discretized equation is applied"""
        return self.x_indicies[
            np.isin(self.x_indicies, self.restricted_indicies, invert=True)
        ]

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
            return res[self.free_indicies]
        else:
            # Assumes independent conditions
            for condition in self.conditions:
                res[condition.m] = condition.solve_restricted(
                    v, self.M + 2, self.h, n * self.k
                )
            return res

    def restrict(self, v):
        """Restrict a vector to self.free_indicies"""
        return v[self.free_indicies]


class HeatEquation(Equation):
    @cached_property
    def restricted_indicies(self):
        return np.array((0, self.M + 1), dtype=np.int64)

    def operator(self):
        return central_difference(self.M + 2, power=2, format="csc") / self.h ** 2


class InviscidBurgers(Equation):
    @cached_property
    def restricted_indicies(self):
        return np.array((0, self.M + 1), dtype=np.int64)

    @cache
    def get_operator(self):
        def operator(v):
            res = np.empty((self.M + 2,), dtype=np.float64)
            res[self.free_indicies] = -v[1:-1] / (2 * self.h) * (v[2:] - v[:-2])
            return res

        return operator


class InviscidBurgers2(InviscidBurgers):
    @cache
    def get_operator(self):
        def operator(v):
            res = np.empty((self.M + 2,), dtype=np.float64)
            res[self.free_indicies] = -(v[2:] ** 2 - v[:-2] ** 2) / (4 * self.h)
            return res

        return operator


class PeriodicKdV(Equation):
    """Linearized Korteweg-deVries with periodic boundary condition with period 2"""

    @cached_property
    def restricted_indicies(self):
        return np.array([self.M + 1], dtype=np.int64)

    def operator(self):
        # Since the equation is on [-1, 1] we introduce a shift in x: 2 * (x-1/2) to
        # solve it on [0, 1] instead. By solving this alternate diff. eqn we introduce a
        # factor 2 for each power of the derivative, so we divide the operator by 2**p.

        d3 = np.zeros((self.M + 1,))
        d3[0:7] = np.array((-1, 0, 3, 0, -3, 0, 1)) / (8 * self.h ** 3) / 2 ** 3
        d1 = np.zeros((self.M + 1,))
        d1[0:7] = np.array((0, 0, -1, 0, 1, 0, 0)) / (2 * self.h) / 2

        single_operator = -d3 - (1 + np.pi ** 2) * d1
        zero_indexed = np.roll(single_operator, -3)

        operator = scipy.sparse.lil_matrix((self.M + 2, self.M + 2), dtype=np.float64)
        for i in range(self.M + 2):
            operator[i, :-1] = np.roll(zero_indexed, i)

        return operator

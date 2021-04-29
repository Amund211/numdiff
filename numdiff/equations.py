import numpy as np
import scipy.sparse.linalg

from .cache import cache, cached_property
from .helpers import central_difference


class Equation:
    """
    Specify the discretization of the operator L, L_h, in `.operator()` as a matrix.

    If the operator isn't linear you can override `.get_operator()`
    """

    periodic = False

    def __init__(self, *, M, conditions=(), **kwargs):
        self.M = M
        self.conditions = conditions
        super().__init__(**kwargs)

        assert self.M > 0

    @cached_property
    def length(self):
        """The length of the solution"""
        return self.M if self.periodic else self.M + 2

    @cached_property
    def h(self):
        return 1 / (self.length - 1 if not self.periodic else self.length)

    @cached_property
    def x_indicies(self):
        """The x indicies that will be solved for x = i * h"""
        return np.arange(0, self.length)

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

    @cached_property
    def single_operator(self):
        """
        The first row of the operator matrix

        Used to create the full matrix when the equation is periodic
        """
        raise NotImplementedError

    @cache
    def operator(self):
        """A matrix representing the discretized operator L_h"""
        if not self.periodic:
            raise NotImplementedError

        values = []
        diags = []
        for i, value in enumerate(self.single_operator):
            if value == 0:
                continue

            values.append(value)
            diags.append(i)

            if i != 0:
                values.append(value)
                diags.append(i - self.length)

        return scipy.sparse.diags(
            values,
            diags,
            shape=(self.length, self.length),
            dtype=np.float64,
            format="dok",
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
            return res[self.free_indicies]
        else:
            # Assumes independent conditions
            for condition in self.conditions:
                res[condition.m] = condition.solve_restricted(
                    v, self.length, self.h, n * self.k
                )
            return res

    def restrict(self, v):
        """Restrict a vector to self.free_indicies"""
        return v[self.free_indicies]


class HeatEquation(Equation):
    @cached_property
    def restricted_indicies(self):
        return np.array((0, self.length - 1), dtype=np.int64)

    @cache
    def operator(self):
        return central_difference(self.length, power=2, format="csc") / self.h ** 2


class InviscidBurgers(Equation):
    @cached_property
    def restricted_indicies(self):
        return np.array((0, self.length - 1), dtype=np.int64)

    @cache
    def get_operator(self):
        def operator(v):
            res = np.empty((self.length,), dtype=np.float64)
            res[self.free_indicies] = -v[1:-1] / (2 * self.h) * (v[2:] - v[:-2])
            return res

        return operator


class InviscidBurgers2(InviscidBurgers):
    @cache
    def get_operator(self):
        def operator(v):
            res = np.empty((self.length,), dtype=np.float64)
            res[self.free_indicies] = -(v[2:] ** 2 - v[:-2] ** 2) / (4 * self.h)
            return res

        return operator


class PeriodicKdV(Equation):
    """Linearized Korteweg-deVries with periodic boundary condition with period 2"""

    periodic = True

    @cached_property
    def restricted_indicies(self):
        return np.array((), dtype=np.int64)

    @cached_property
    def single_operator(self):
        # Since the equation is on [-1, 1] we introduce a shift in x: 2 * (x-1/2) to
        # solve it on [0, 1] instead. By solving this alternate diff. eqn we introduce a
        # factor 2 for each power of the derivative, so we divide the operator by 2**p.

        d3 = np.zeros((self.length,))
        d3[0:7] = np.array((-1, 0, 3, 0, -3, 0, 1)) / (8 * self.h ** 3) / 2 ** 3
        d1 = np.zeros((self.length,))
        d1[0:7] = np.array((0, 0, -1, 0, 1, 0, 0)) / (2 * self.h) / 2

        single_operator = -d3 - (1 + np.pi ** 2) * d1
        return np.roll(single_operator, -3)


class _AdvectionDiffusionBase(Equation):
    def __init__(self, *, c, d, **kwargs):
        self.c = c
        self.d = d

        assert self.c > 0, "c must be positive"
        assert self.d > 0, "d must be positive"

        super().__init__(**kwargs)


class PeriodicAdvectionDiffusion1stOrder(_AdvectionDiffusionBase):
    """
    The advection-diffusion equation with periodic boundary condition with period 1

    Both the first and second derivative have order 1 (forward difference)
    """

    periodic = True

    @cached_property
    def restricted_indicies(self):
        return np.array((), dtype=np.int64)

    @cached_property
    def single_operator(self):
        # The first forward difference
        d1 = np.zeros((self.length,))
        d1[:2] = np.array((-1, 1)) / self.h

        # The second forward difference
        d2 = np.zeros((self.length,))
        d2[:3] = np.array((1, -2, 1)) / self.h ** 2

        return self.c * d1 + self.d * d2


class PeriodicAdvectionDiffusion2ndOrder(_AdvectionDiffusionBase):
    """
    The advection-diffusion equation with periodic boundary condition with period 1

    Both the first and second derivative have order 2
    """

    periodic = True

    @cached_property
    def restricted_indicies(self):
        return np.array((), dtype=np.int64)

    @cached_property
    def single_operator(self):
        # The first derivative finite difference
        d1 = np.zeros((self.length,))
        d1[:3] = np.array((-1, 0, 1)) / (2 * self.h)

        # The second derivative finite difference
        d2 = np.zeros((self.length,))
        d2[:3] = np.array((1, -2, 1)) / self.h ** 2

        single_operator = self.c * d1 + self.d * d2
        return np.roll(single_operator, -1)


class PeriodicAdvectionDiffusion4thOrder(_AdvectionDiffusionBase):
    """
    The advection-diffusion equation with periodic boundary condition with period 1

    Both the first and second derivative have order 4
    """

    periodic = True

    @cached_property
    def restricted_indicies(self):
        return np.array((), dtype=np.int64)

    @cached_property
    def single_operator(self):
        # The first derivative finite difference
        d1 = np.zeros((self.length,))
        d1[:5] = np.array((1, -8, 0, 8, -1)) / (12 * self.h)

        # The second derivative finite difference
        d2 = np.zeros((self.length,))
        d2[:5] = np.array((-1, 16, -30, 16, -1)) / (12 * self.h ** 2)

        single_operator = self.c * d1 + self.d * d2
        return np.roll(single_operator, -2)


class AdvectionDiffusion2ndOrder(_AdvectionDiffusionBase):
    """
    The advection-diffusion equation

    Both the first and second derivative have order 2
    """

    @cached_property
    def restricted_indicies(self):
        return np.array((0, self.length - 1), dtype=np.int64)

    @cache
    def operator(self):
        return (
            self.c * central_difference(self.length, power=1, format="dok") / self.h
            + self.d
            * central_difference(self.length, power=2, format="dok")
            / self.h ** 2
        )

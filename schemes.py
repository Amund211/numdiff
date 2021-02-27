# https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

import numpy as np
import scipy.sparse.linalg

from conditions import Condition, Neumann


def central_difference(N, power=2):
    diag = -2 * np.ones(N)
    offdiag = np.ones(N - 1)

    return (np.diag(diag) + np.diag(offdiag, k=-1) + np.diag(offdiag, k=1)).astype(
        np.float64
    )


def poisson(f, M, condition_1, condition_2):
    assert M >= 4

    h = 1 / (M + 1)
    A = central_difference(M + 2, power=2) / h ** 2

    # x1=h, x2=2h, ..., xm+1 = 1
    x = np.arange(0, M + 2).astype(np.float64) * h
    f = f(x)

    A[0, :] = condition_1.get_vector(length=M + 2, h=h)
    f[0] = condition_1.get_scalar()

    A[-1, :] = condition_2.get_vector(length=M + 2, h=h)
    f[-1] = condition_2.get_scalar()

    U = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), f)

    return x, U


def central_difference_operator(context, m, n, power=2):
    if power == 1:
        return context[m + 1, n] - context[m - 1, n]
    elif power == 2:
        return context[m + 1, n] - 2 * context[m, n] + context[m - 1, n]

    raise ValueError


@dataclass(frozen=True)
class Scheme:
    M: int
    N: int
    k: float

    conditions: Iterable[Condition]

    # Whether the step matrix should be factorized, set to False if it often changes
    factorize = True

    def __post_init__(self):
        assert self.k >= 0
        assert self.h >= 0
        self.validate_r()

    @property
    def h(self):
        return 1 / (self.M + 1)

    @property
    def r(self):
        return self.k / self.h ** 2

    @property
    def free_indicies(self):
        """
        np.array of indicies that are not calculated by the scheme
        These are free to be used as conditions
        """
        raise NotImplementedError

    @property
    def restricted_x_indicies(self):
        """Get the x axis indicies where this method has all its needed context"""
        unrestricted = np.arange(0, self.M + 2)
        return unrestricted[np.isin(unrestricted, self.free_indicies, invert=True)]

    def validate_r(self):
        """Validate that the method is convergent with the given r"""
        raise NotImplementedError

    def rhs(self, context, n):
        raise NotImplementedError

    def matrix(self):
        raise NotImplementedError

    def operator(self):
        raise NotImplementedError

    def get_constrained_rhs(self, context, n):
        b = self.rhs(context, n)
        for condition, index in zip(self.conditions, self.free_indicies):
            scalar = condition.get_scalar(n * self.k)
            b[index] = scalar

        return b

    @cache
    def get_constrained_matrix(self):
        assert len(self.conditions) == len(
            self.free_indicies
        ), f"Must have exactly  {len(self.free_indicies)} boundary conditions"

        A = self.matrix()

        for condition, index in zip(self.conditions, self.free_indicies):
            eqn = condition.get_vector(length=self.M + 2, h=self.h)
            A[index, :] = eqn

        return scipy.sparse.csc_matrix(A)

    @cache
    def get_solver(self):
        sparse = self.get_constrained_matrix()
        if self.factorize:
            return scipy.sparse.linalg.factorized(sparse)
        else:
            return lambda b: scipy.sparse.linalg.spsolve(sparse, b)

    def step(self, context, n):
        return self.get_solver()(self.get_constrained_rhs(context, n))

    def apply_operator(self, context, n):
        """Apply the discretized operator in x to the values from timestep n"""
        rhs = self.operator() @ context[:, n]

        return rhs[self.restricted_x_indicies]


def euler_scheme(context, m, n, r):
    # Return the rhs in the system of eqn to solve for x_m^n
    # For euler, the matrix is the identity
    return context[m, n] + r * central_difference_operator(context, m, n, power=2)


class Euler(Scheme):
    @property
    def free_indicies(self):
        return np.array((0, self.M + 1), dtype=np.int64)

    def validate_r(self):
        assert self.r <= 1 / 2, f"r <= 1/2 <= {self.r} needed for convergence"

    def rhs(self, context, n):
        rhs = np.empty((self.M + 2,), dtype=np.float64)
        rhs[self.free_indicies] = 0

        rhs[self.restricted_x_indicies] = context[
            self.restricted_x_indicies, n - 1
        ] + self.k * self.apply_operator(context, n - 1)
        return rhs

    def matrix(self):
        # Euler is explicit, so no need to solve a system => identity
        return np.eye(self.M + 2)

    def operator(self):
        return central_difference(self.M + 2, power=2) / self.h ** 2


@dataclass(frozen=True)
class ThetaMethod(Scheme):
    """
    The theta method
    theta=0 => euler
    theta=1/2 => CN
    theta=1 => implicit euler
    """

    theta: float

    @property
    def free_indicies(self):
        return np.array((0, self.M + 1), dtype=np.int64)

    def validate_r(self):
        if 1 / 2 <= self.theta <= 1:
            # All r >= 0
            return
        elif 0 <= self.theta < 1 / 2:
            eta = 0
            for condition in self.conditions:
                if isinstance(condition, Neumann):
                    eta = max(eta, abs(condition.condition))

            assert self.r <= 1 / (
                2 * (1 - 2 * self.theta) * (1 + eta * self.h / 2)
            ), f"r <= 1/(2(1-2theta)(1+eta*h/2)) <= {self.r} needed for convergence"
        else:
            raise ValueError(f"Invalid value for theta {self.theta}")

    def rhs(self, context, n):
        rhs = np.empty((self.M + 2,), dtype=np.float64)
        rhs[self.free_indicies] = 0

        rhs[self.restricted_x_indicies] = context[self.restricted_x_indicies, n - 1] + (
            1 - self.theta
        ) * self.k * self.apply_operator(context, n - 1)

        return rhs

    def matrix(self):
        return np.eye(self.M + 2) - self.theta * self.k * self.operator()

    def operator(self):
        return central_difference(self.M + 2, power=2) / self.h ** 2


def solve_time_evolution(scheme, f):
    sol = np.empty((scheme.M + 2, scheme.N + 1), dtype=np.float64)
    x_axis = np.linspace(0, 1, scheme.M + 2)
    sol[:, 0] = f(x_axis)

    for n in range(1, scheme.N + 1):
        U = scheme.step(sol, n)

        sol[:, n] = U

    return x_axis, sol

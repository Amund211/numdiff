# https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache, cached_property

import numpy as np
import scipy.sparse.linalg

from conditions import Condition, Neumann


@dataclass(frozen=True)
class Scheme:
    """
    A general scheme class for linear 1D time evolution equations u_t = Lu

    Specify L by inheriting from an `Equation`
    Specify the time discretization in `.matrix()` and `.rhs()`.

    If the time discretization is explicit, the operator need not be linear.

    Uses a direct method for solving the linear system, override `.get_solver()` to use
    an iterative method. (ex.: scipy.sparse.linalg.cg)
    """

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

    @cached_property
    def h(self):
        return 1 / (self.M + 1)

    @cached_property
    def r(self):
        return self.k / self.h ** 2

    def validate_r(self):
        """Validate that the method is convergent with the given r"""

        raise NotImplementedError

    def matrix(self):
        """
        A matrix representing the left hand side (t_n+1) of the discretized system
        in time.
        """

        raise NotImplementedError

    def rhs(self, context, n):
        """
        The right hand side (t_n) corresponding to the time discretized system given
        by `.matrix()`
        """

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

    def solve(self, f):
        """
        Solve the time evolution equation
        """

        sol = np.empty((self.M + 2, self.N + 1), dtype=np.float64)
        x_axis = np.linspace(0, 1, self.M + 2)
        sol[:, 0] = f(x_axis)

        for n in range(1, self.N + 1):
            U = self.step(sol, n)

            sol[:, n] = U

        return x_axis, sol


class Euler(Scheme):
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


@dataclass(frozen=True)
class ThetaMethod(Scheme):
    """
    The theta method
    theta=0 => euler
    theta=1/2 => CN
    theta=1 => implicit euler
    """

    theta: float

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

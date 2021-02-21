# https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Any, List, Optional, Union

import numpy as np


def central_difference(N, order=2):
    diag = -2 * np.ones(N)
    offdiag = np.ones(N - 1)

    return np.diag(diag) + np.diag(offdiag, k=-1) + np.diag(offdiag, k=1)


def poisson(f, M, alpha, sigma):
    assert M >= 4

    h = 1 / (M + 1)
    A = central_difference(M + 1, order=2) / h ** 2

    # x1=h, x2=2h, ..., xm+1 = 1
    x = np.arange(1, M + 2) * h
    f = f(x)

    # Dirichlet
    f[0] -= alpha / h ** 2

    # Adjust for neumann in right endpoint
    # A[-1, -3:] = (-1/(2*h), 2/h, - 3/(2*h))
    # Provided schema has wrong signs
    A[-1, -3:] = (1 / (2 * h), -2 / h, 3 / (2 * h))
    f[-1] = sigma

    U = np.linalg.solve(A, f)

    return x, U


def euler_scheme(context, m, n, r):
    # Return the rhs in the system of eqn to solve for x_m^n
    # For euler, the matrix is the identity
    return context[m][n] + r * central_difference_operator(context, m, n, order=2)


def central_difference_operator(context, m, n, order=2):
    if order == 1:
        return context[m + 1][n] - context[m - 1][n]
    elif order == 2:
        return context[m + 1][n] - 2 * context[m][n] + context[m - 1][n]

    raise ValueError


@dataclass
class Condition:
    condition: Any  # Union[float, Callable[[int], float]]
    m: int

    def get_condition_value(self, n):
        return self.condition(n) if callable(self.condition) else self.condition

    def get_equation(self, n, M, h):
        raise NotImplementedError


class Dirichlet(Condition):
    def get_equation(self, n, M, h):
        cond_value = self.get_condition_value(n)
        lhs = np.zeros(M + 2)
        lhs[self.m] = 1
        rhs = cond_value
        return lhs, rhs


@dataclass
class Neumann(Condition):
    order: Optional[int] = None  # Order of the neumann condition

    def get_equation(self, n, M, h):
        cond_value = self.get_condition_value(n)
        lhs = np.zeros(M + 2)
        if self.m == 0:
            if self.order == 1:
                lhs[0:2] = np.array((-1, 1)) / h ** 2
                rhs = h * cond_value
            elif self.order == 2:
                lhs[0:3] = np.array((-3 / 2, 2, -1 / 2)) / h ** 2
                rhs = h * cond_value
            else:
                raise ValueError
        elif self.m == M + 1:
            if self.order == 1:
                lhs[-2:] = np.array((-1, 1)) / h ** 2
                rhs = h * cond_value
            elif self.order == 2:
                lhs[-3:] = np.array((1 / 2, -2, 3 / 2)) / h ** 2
                rhs = h * cond_value
            else:
                raise ValueError
        else:
            # Order 2
            lhs[self.m + 1] = 1
            lhs[self.m - 1] = -1
            rhs = cond_value
        return lhs, rhs


def apply_conditions(self, A, b, conditions, indicies):
    if left_boundary.dirichlet:
        matrix[0][0] = 1
        rhs[0] = left_boundary.get_dirichlet(n)

    if left_boundary.neumann:
        matrix[0][0] = 1
        rhs[0] = left_boundary.get_neumann(n)


@dataclass
class Scheme:
    M: int
    N: int

    conditions = Iterable[Condition]

    @property
    def h(self):
        return 1 / (self.M + 1)

    @property
    def k(self):
        return 1 / self.N

    @property
    def r(self):
        return self.k / self.h ** 2

    @property
    def free_indicies(self):
        """
        Indicies that are not calculated by the scheme
        These are free to be used as conditions
        """
        raise NotImplementedError

    @property
    def restricted_x_indicies(self):
        """Get the x axis indicies where this method has all its needed context"""
        unrestricted = np.arange(0, self.M + 2)
        return unrestricted[np.isin(unrestricted, self.free_indicies)]

    def rhs(self, context, n):
        raise NotImplementedError

    def matrix(self):
        raise NotImplementedError

    def apply_conditions(self, A, b, n):
        assert len(self.conditions) == len(
            self.free_indicies
        ), f"Must have exactly  {len(self.free_indicies)} boundary conditions"

        for condition, index in zip(self.conditions, self.free_indicies):
            lhs, rhs = condition.get_equation(n, self.M, self.h)
            A[index] = lhs
            b[index] = rhs

        return A, b

    def system(self, context, n, r):
        matrix = self.matrix()
        rhs = self.rhs(context, n, r)

        matrix, rhs = self.apply_conditions(matrix, rhs, n)

        return matrix, rhs


class Euler(Scheme):
    @property
    def free_indicies(self):
        return (0, self.M + 1)

    def rhs(self, context, n):
        rhs = np.empty((self.M + 2,))
        rhs[self.free_indicies] = 0
        rhs[1 : self.M + 1] = context[self.restricted_x_indicies][
            n - 1
        ] + self.r * central_difference_operator(
            context, self.restricted_x_indicies, n - 1, order=2
        )

        return rhs

    def rhs_scalar(self, context, m, n):
        if m == 0 or m == self.M + 1:
            return 0
        return context[m][n - 1] + self.r * central_difference_operator(
            context, m, n - 1, order=2
        )

    def matrix(self):
        # Euler is explicit, so no need to solve a system => identity
        return np.eye(self.M + 2)


def solve_time_evolution(scheme):
    sol = np.empty((M + 2, N))
    x_axis = np.linspace(0, 1, scheme.M + 2)
    sol[:, 0] = f(x_axis)

    for n in range(1, N):
        # M+2-2 bc two dirichlet boundary cond. g0 and g1
        A = scheme.matrix()

        ms = np.arange(1, M + 1)
        rhs = scheme.rhs(sol, ms, n, r)

        U = np.linalg.solve(A, rhs)

        sol[:][n] = U

    return x_axis * h, sol


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def f(x):
        return np.cos(2 * np.pi * x) + x

    M = 9
    alpha = 0
    sigma = 1

    def u(x):
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (sigma - 1 / 2) * x
            + alpha
        )

    x, U = poisson(f, M, alpha, sigma)

    plt.plot(x, U, label="U")
    plt.plot(x, u(x), label="u")
    plt.legend()
    plt.show()

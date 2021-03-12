from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Condition:
    """
    An additional condition on a differential equation represented as a linear equation

    NOTE: The vector returned from `.get_vector` must be constant
    """

    m: int

    def get_condition_value(self, t=None):
        return self.condition(t) if callable(self.condition) else self.condition

    def get_vector(self, length, h):
        """Vector representing the equation"""
        raise NotImplementedError

    def get_scalar(self, t=None):
        """Scalar representing the rhs"""
        # Default implementation
        return self.get_condition_value(t)

    def solve_restricted(self, v, length, h, t):
        """Given a vector, solve for the value at `self.m`"""

        restriction = self.get_vector(length, h=h)
        weight = restriction[self.m]
        assert weight != 0, "Cannot solve for this index when its weight is 0"

        restriction[self.m] = 0
        lhs = np.dot(restriction, v)
        rhs = self.get_scalar(t)

        return (rhs - lhs) / weight


@dataclass(frozen=True)
class Dirichlet(Condition):
    condition: Any  # Union[float, Callable[[float], float]]

    def get_vector(self, length, **kwargs):
        eqn = np.zeros(length)
        eqn[self.m] = 1
        return eqn


@dataclass(frozen=True)
class Neumann(Condition):
    condition: Any  # Union[float, Callable[[float], float]]
    order: int = 2  # Order of the neumann condition

    def get_vector(self, length, h, **kwargs):
        eqn = np.zeros(length)
        if self.m == 0:
            if self.order == 1:
                eqn[0:2] = np.array((-1, 1)) / h
            elif self.order == 2:
                eqn[0:3] = np.array((-3 / 2, 2, -1 / 2)) / h
            else:
                raise ValueError
        elif self.m == -1 or self.m == length - 1:  # Last index
            if self.order == 1:
                eqn[-2:] = np.array((-1, 1)) / h
            elif self.order == 2:
                eqn[-3:] = np.array((1 / 2, -2, 3 / 2)) / h
            else:
                raise ValueError
        else:
            # Order 2
            eqn[self.m + 1] = 1 / h
            eqn[self.m - 1] = -1 / h
        return eqn


@dataclass(frozen=True)
class Periodic(Condition):
    period: int

    def get_vector(self, length, **kwargs):
        eqn = np.zeros(length)
        eqn[self.m] = 1
        eqn[self.m + self.period] = -1
        return eqn

    def get_scalar(self, t=None):
        return 0

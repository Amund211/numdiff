from dataclasses import dataclass
from typing import Any

import numpy as np


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
    order: int = 2  # Order of the neumann condition

    def get_equation(self, n, M, h):
        cond_value = self.get_condition_value(n)
        rhs = cond_value / h
        lhs = np.zeros(M + 2)
        if self.m == 0:
            if self.order == 1:
                lhs[0:2] = np.array((-1, 1)) / h ** 2
            elif self.order == 2:
                lhs[0:3] = np.array((-3 / 2, 2, -1 / 2)) / h ** 2
            else:
                raise ValueError
        elif self.m == M + 1:
            if self.order == 1:
                lhs[-2:] = np.array((-1, 1)) / h ** 2
            elif self.order == 2:
                lhs[-3:] = np.array((1 / 2, -2, 3 / 2)) / h ** 2
            else:
                raise ValueError
        else:
            # Order 2
            lhs[self.m + 1] = 1 / h ** 2
            lhs[self.m - 1] = -1 / h ** 2
        return lhs, rhs

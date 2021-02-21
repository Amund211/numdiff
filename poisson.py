# https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf
import numpy as np

def central_difference(N, order=2):
    diag = -2 * np.ones(N)
    offdiag = np.ones(N - 1)

    return np.diag(diag) + np.diag(offdiag, k=-1) + np.diag(offdiag, k=1)


from dataclasses import dataclass
from typing import Optional, List

class BoundaryConditions(dataclass):
    dirichlet: Optional[float] = None
    neumann: Optional[float] = None


    def apply_conditions(self, scheme):
        pass


def poisson(f, M, alpha, sigma):
    assert M >= 4

    h = 1/(M+1)
    A = central_difference(M+1, order=2) / h**2

    # x1=h, x2=2h, ..., xm+1 = 1
    x = np.arange(1, M + 2) * h
    f = f(x)


    # Dirichlet
    f[0] -= alpha / h**2

    # Adjust for neumann in right endpoint
    #A[-1, -3:] = (-1/(2*h), 2/h, - 3/(2*h))
    # Provided schema has wrong signs
    A[-1, -3:] = (1/(2*h), -2/h, 3/(2*h))
    f[-1] = sigma

    U = np.linalg.solve(A, f)

    return x, U


def euler_scheme(context, m, n, r):
    # Return the rhs in the system of eqn to solve for x_m^n
    # For euler, the matrix is the identity
    return context[m][n] + r * central_difference_operator(context, m, n, order=2)

def central_difference_operator(context, m, n, order=2):
    if order == 1:
        return context[m+1][n] - context[m-1][n]
    elif order == 2:
        return context[m+1][n] - 2 * context[m][n] + context[m-1][n]

    raise ValueError


class Scheme(dataclass):
    M: int
    N: int

    left_boundary?

    def rhs(context, m, n, r):
        raise NotImplementedError

    def matrix(context, m, n):
        raise NotImplementedError

class Euler(Scheme):
    def rhs(self, context, m, n, r):
        return context[m][n-1] + r * central_difference_operator(context, m, n-1, order=2)

    def matrix(self):
        # M+2-2 bc two dirichlet boundary cond. g0 and g1
        return np.eye(M)


def solve_time_evolution(scheme):
    sol = np.empty((M+2, N))
    x_axis = np.linspace(0, 1, scheme.M+2)
    sol[:, 0] = f(x_axis)
    h = 1/(scheme.M+1)
    k = 1/scheme.N

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
        return (1 / (2 * np.pi)**2) * (1-np.cos(2*np.pi*x)) + x**3/6 + (sigma - 1/2)*x + alpha

    x, U = poisson(f, M, alpha, sigma)

    plt.plot(x, U, label="U")
    plt.plot(x, u(x), label="u")
    plt.legend()
    plt.show()



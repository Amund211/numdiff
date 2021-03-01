import matplotlib.pyplot as plt
import numpy as np

from conditions import Dirichlet, Neumann
from poisson import poisson
from schemes import Euler, ThetaMethod, solve_time_evolution


def test_poisson():
    def f(x):
        return np.cos(2 * np.pi * x) + x

    M = 10000
    alpha = 0
    sigma = 1

    def u(x):
        # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
        # Here: solved for left dirichlet and right neumann
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (sigma - 1 / 2) * x
            + alpha
        )

    x, U = poisson(
        f=f,
        M=M,
        condition_1=Dirichlet(condition=alpha, m=0),
        condition_2=Neumann(condition=sigma, m=M + 1),
    )

    plt.plot(x, U, label="U")
    plt.plot(x, u(x), label="u")
    plt.legend()
    plt.show()


def test_heat_euler():
    M = 100
    N = 2000
    k = 1 / (M + 2) ** 2 / 2.5

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    # def f(x):
    # return 2 * x * (x < 1 / 2) + (2 - 2 * x) * (x >= 1 / 2)

    scheme = Euler(
        M=M,
        N=N,
        k=k,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=M + 1)),
        # conditions=(Neumann(condition=0, m=0), Dirichlet(condition=0, m=M + 1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=M+1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=M+1)),
    )

    x_axis, sol = solve_time_evolution(scheme, f)

    for n in range(0, N + 1, max(N // 10, 1)):
        plt.plot(x_axis, sol[:, n], label=f"U(t={n*k:.3f}, n={n})")
    plt.legend()
    plt.show()


def test_heat_theta():
    M = 10000
    N = 2000
    # k = 1 / (M + 2) ** 2 / 2.5
    k = 0.001
    theta = 1 / 2

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    # def f(x):
    # return 2 * x * (x < 1 / 2) + (2 - 2 * x) * (x >= 1 / 2)

    scheme = ThetaMethod(
        M=M,
        N=N,
        k=k,
        theta=theta,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=M + 1)),
        # conditions=(Neumann(condition=0, m=0), Dirichlet(condition=0, m=M + 1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=M+1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=M+1)),
    )

    x_axis, sol = solve_time_evolution(scheme, f)

    for n in range(0, N + 1, max(N // 10, 1)):
        plt.plot(x_axis, sol[:, n], label=f"U(t={n*k:.3f}, n={n})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_poisson()
    # test_heat_euler()
    # test_heat_theta()

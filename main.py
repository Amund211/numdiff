import matplotlib.pyplot as plt
import numpy as np

from conditions import Dirichlet, Neumann
from equations import HeatEquation, InviscidBurgers, InviscidBurgers2, PeriodicKdV
from poisson import poisson
from schemes import RK4, Euler, ThetaMethod


def solve_and_plot(scheme, f):
    x_axis, sol = scheme.solve(f)

    for n in range(0, scheme.N + 1, max(scheme.N // 10, 1)):
        plt.plot(x_axis, sol[:, n], label=f"U(t={n*scheme.k:.3f}, n={n})")
    plt.legend()
    plt.show()


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
        # condition_1=Neumann(condition=sigma - 1 / 2, m=0),
        condition_2=Neumann(condition=sigma, m=M + 1),
        maxiter=1e6,
        explain_solution=True,
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

    class HeatEuler(Euler, HeatEquation):
        pass

    scheme = HeatEuler(
        M=M,
        N=N,
        k=k,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=M + 1)),
        # conditions=(Neumann(condition=0, m=0), Dirichlet(condition=0, m=M + 1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=M+1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=M+1)),
    )

    solve_and_plot(scheme, f)


def test_heat_theta():
    M = 10000
    N = 2000
    k = 3.85e-05
    theta = 1 / 2

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    # def f(x):
    # return 2 * x * (x < 1 / 2) + (2 - 2 * x) * (x >= 1 / 2)

    class HeatTheta(ThetaMethod, HeatEquation):
        pass

    scheme = HeatTheta(
        M=M,
        N=N,
        k=k,
        theta=theta,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=M + 1)),
        # conditions=(Neumann(condition=0, m=0), Dirichlet(condition=0, m=M + 1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=M+1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=M+1)),
    )

    solve_and_plot(scheme, f)


def test_heat_rk4():
    M = 1000
    N = 2000
    N = 200
    k = 3.85e-05

    M = 100
    N = 200000
    k = 1 / (M + 2) ** 2 / 2.5

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    class HeatRK4(RK4, HeatEquation):
        pass

    scheme = HeatRK4(
        M=M,
        N=N,
        k=k,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=M + 1)),
        # conditions=(Neumann(condition=0, m=0), Dirichlet(condition=0, m=M + 1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=M+1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=M+1)),
    )

    solve_and_plot(scheme, f)


def test_burgers_rk4():
    M = 1000
    N = 2000
    N = 200
    k = 3.85e-05

    M = 1000
    N = 200000
    k = 1 / (M + 2) ** 2 / 2.5

    def f(x):
        return np.exp(-400 * (x - 1 / 2) ** 2)

    class BurgersRK4(RK4, InviscidBurgers):
        pass

    scheme = BurgersRK4(
        M=M,
        N=N,
        k=k,
        conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=M + 1)),
    )

    solve_and_plot(scheme, f)


def test_KdV():
    M = 1000
    N = 2000
    N = 20000
    k = 1e-3

    # M = 1000
    # N = 200000
    # k = 1 / (M + 2) ** 2 / 2.5

    def f(x):
        return np.sin(np.pi * (2 * (x - 1 / 2)))

    class KdVTheta(ThetaMethod, PeriodicKdV):
        pass

    scheme = KdVTheta(
        M=M - 1,
        N=N,
        k=k,
        conditions=(),
        theta=1 / 2,  # 1/2 => CN
    )

    solve_and_plot(scheme, f)


if __name__ == "__main__":
    # test_poisson()
    # test_heat_euler()
    # test_heat_theta()
    # test_heat_rk4()
    # test_burgers_rk4()
    test_KdV()

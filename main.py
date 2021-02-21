import numpy as np


def test_poisson():
    import matplotlib.pyplot as plt

    from schemes import poisson

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


def test_heat_euler():
    import matplotlib.pyplot as plt

    from conditions import Dirichlet, Neumann
    from schemes import Euler, solve_time_evolution

    M = 80
    N = 1000
    k = 0.001

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    # scheme = Euler(M=M, N=N, k=k, conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=M+1)))
    scheme = Euler(
        M=M,
        N=N,
        k=k,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=M + 1)),
    )
    x_axis, sol = solve_time_evolution(scheme, f)

    for n in range(0, N + 1, 50):
        plt.plot(x_axis, sol[:, n], label=f"U({n})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_poisson()
    test_heat_euler()

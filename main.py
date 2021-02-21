import numpy as np

from schemes import poisson

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

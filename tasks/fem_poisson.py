import matplotlib.pyplot as plt
import numpy as np

from fem import AFEM, FEM_error_plot


def f(x):
    return -2


def u(x):
    return x ** 2


a = 0
b = 1
d1 = 0
d2 = 1


def task_5b_refinement():
    N_array = np.power(2, np.arange(3, 12), dtype=np.int32)

    FEM_error_plot(N_array, a, b, f, d1, d2, u)


def task_5b_avg():
    N = 20
    alpha1 = 1
    estimate1 = "averaging"

    X1, U1 = AFEM(N, f, u, a, b, d1, d2, alpha1, estimate1)
    plt.plot(X1, u(X1), ".", color="hotpink")
    plt.plot(X1, U1, color="aqua")
    plt.grid()
    plt.title(len(X1))


def task_5b_max():
    N = 20
    alpha2 = 0.7
    estimate2 = "maximum"
    X2, U2 = AFEM(N, f, u, a, b, d1, d2, alpha2, estimate2)
    plt.plot(X2, u(X2), ".", color="hotpink")
    plt.plot(X2, U2, color="aqua")
    plt.title(len(X2))
    plt.grid()

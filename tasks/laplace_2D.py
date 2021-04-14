import matplotlib.pyplot as plt
import numpy as np

from laplace import Laplace, analytical, d_norm


def task_3bx():
    M = np.array([5, 10, 30, 50, 100])
    error = np.zeros(len(M))
    for i, m in enumerate(M):
        U = Laplace(m, m)
        u, X = analytical(m, m)
        error[i] = d_norm(u - U) / d_norm(u)

    plt.plot(M ** 2, error)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of grid-points MxMy")
    plt.ylabel("Relative error in discrete norm")
    plt.grid()

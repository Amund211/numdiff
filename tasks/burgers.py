import matplotlib.pyplot as plt
import numpy as np

from numdiff.conditions import Dirichlet
from numdiff.equations import InviscidBurgers
from numdiff.schemes import RK4
from numdiff.settings import FINE_PARAMETERS


class BurgersRK4(RK4, InviscidBurgers):
    pass


def task_2c():
    if FINE_PARAMETERS:
        M = 80000
        N = 20000
    else:
        M = 2000
        N = 2000

    max_T = 0.059

    k = max_T / N

    t_range = np.linspace(0.0576, 0.0584, 10)

    min_t = t_range[0]
    min_t_index = int(min_t / k)
    context = N - min_t_index + 1

    def f(x):
        return np.exp(-400 * (x - 1 / 2) ** 2)

    scheme = BurgersRK4(
        M=M,
        N=N,
        k=k,
        conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=-1)),
    )

    x_axis, sol = scheme.solve(f, context=context)

    for i, t in enumerate(t_range):
        n = int(t / k)

        plt.plot(
            x_axis,
            sol[:, n].view(np.ndarray),
            label=f"$U(x, t={n*scheme.k:.5f})$",
            color=f"C{i % 10}",
        )

    plt.legend()
    plt.xlim(0.5695, 0.5711)
    plt.ylim(0.3, 0.9)

    plt.suptitle("The inviscid Burgers' equation - numerical breaking of the wavefront")
    plt.title(f"$M={M}, N={N}$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid()

    plt.legend()

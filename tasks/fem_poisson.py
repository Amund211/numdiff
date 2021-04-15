import matplotlib.pyplot as plt
import numpy as np

from fem import AFEM
from refine import refine_mesh
from refinement_utilities import calculate_relative_L2_error_FEM, make_FEM_solver


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

    ndofs, (distances,) = refine_mesh(
        solver=make_FEM_solver(
            a=a,
            b=b,
            f=f,
            d1=d1,
            d2=d2,
            deg=10,
        ),
        param_range=N_array,
        analytical=u,
        calculate_distances=(calculate_relative_L2_error_FEM,),
    )

    plt.loglog(ndofs, distances, label="$e^r_{L_2}$")

    plt.plot(
        ndofs,
        np.divide(1, (ndofs.astype(np.float64) + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.suptitle("Poisson's equation FEM")
    plt.title(fr"$u\left( x \right) = x^2, x \in \left[ {a}, {b} \right] $")
    plt.xlabel("Degrees of freedom $N-1$")
    plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()


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

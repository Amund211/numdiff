from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from equations import PeriodicKdV
from refine import refine_mesh
from refinement_utilities import calculate_relative_l2_error, make_scheme_solver
from schemes import ThetaMethod


class KdVTheta(ThetaMethod, PeriodicKdV):
    pass


def transform_x(x):
    return 2 * (x - 1 / 2)


def f(x):
    return np.sin(np.pi * transform_x(x))


def u(t, x):
    return np.sin(np.pi * (transform_x(x) - t))


def task_4b():
    T = 1
    N = 10 ** 4

    analytical = partial(u, T)

    scheme_kwargs = {"N": N}

    M_range = np.unique(np.logspace(np.log10(7), 3, num=10, dtype=np.int32))
    # M_range = np.unique(np.logspace(np.log10(7), 4, num=50, dtype=np.int32))
    M_range_euler = np.unique(np.logspace(np.log10(7), 1.5, num=50, dtype=np.int32))

    # Forward Euler
    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(
            cls=KdVTheta,
            f=f,
            T=T,
            scheme_kwargs={"theta": 0, **scheme_kwargs},
        ),
        param_range=M_range_euler,
        analytical=analytical,
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs / N, distances, label="FE")

    # Crank Nicholson
    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(
            cls=KdVTheta,
            f=f,
            T=T,
            scheme_kwargs={"theta": 1 / 2, **scheme_kwargs},
        ),
        param_range=M_range,
        analytical=analytical,
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs / N, distances, label="CN")

    # O(h^2)
    plt.plot(
        M_range,
        1e3 * np.divide(1, M_range ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.grid()

    plt.suptitle("Linearized Korteweg-deVries")
    plt.title(f"Forward Euler vs Crank Nicholson with $N={N}$ at $t={T}$")
    plt.xlabel("Unique spatial nodes $M$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.ylim(ymin=1e-6, ymax=1e2)
    plt.legend()

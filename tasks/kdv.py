from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from numdiff.equations import PeriodicKdV
from numdiff.helpers import l2
from numdiff.refine import refine_mesh
from numdiff.refinement_utilities import calculate_relative_l2_error, make_scheme_solver
from numdiff.schemes import ThetaMethod
from numdiff.settings import FINE_PARAMETERS


class KdVTheta(ThetaMethod, PeriodicKdV):
    pass


def transform_x(x):
    return 2 * (x - 1 / 2)


def f(x):
    return np.sin(np.pi * transform_x(x))


def u(t, x):
    return np.sin(np.pi * (transform_x(x) - t))


def task_4_solution():
    M = 100
    N = 200

    T = 2
    theta = 1 / 2

    k = T / N

    scheme_kwargs = {
        "theta": theta,
        "conditions": (),
        "N": N,
        "k": k,
        "M": M,
    }

    scheme = KdVTheta(**scheme_kwargs)
    x, solution = scheme.solve(f)

    t, x = np.meshgrid(np.linspace(0, T, N + 1), transform_x(np.append(x, 1)))

    c = plt.pcolormesh(
        x, t, np.row_stack((solution, solution[0])), cmap="hot", shading="nearest"
    )
    plt.colorbar(c, ax=plt.gca())

    plt.suptitle(
        f"Linearized Korteweg-deVries - numerical solution with $M={M}, N={N}$"
    )
    plt.title(
        r"$u_t + \left(1 + \pi^2\right) u_x + u_{xxx} = 0, "
        r"u(x, 0) = \sin{\left( \pi x\right)}, u(x, t) = u(x+2, t)$"
    )
    plt.xlabel("$x$")
    plt.ylabel("Time $t$")


def task_4b():
    M_range_euler = np.unique(np.logspace(np.log10(7), 1.5, num=50, dtype=np.int32))

    if FINE_PARAMETERS:
        N = 10 ** 4
        M_range = np.unique(np.logspace(np.log10(7), 4, num=50, dtype=np.int32))
    else:
        N = 10 ** 3
        M_range = np.unique(np.logspace(np.log10(7), 3, num=10, dtype=np.int32))

    T = 1
    analytical = partial(u, T)

    scheme_kwargs = {"N": N}

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


def task_4c():
    if FINE_PARAMETERS:
        N = 10 ** 5
        M = 10 ** 3
    else:
        N = 10 ** 3
        M = 10 ** 3

    T = 10
    k = T / N

    t_axis = np.arange(N + 1) * k

    # Crank Nicolson
    cn = KdVTheta(k=k, N=N, M=M, theta=1 / 2)
    x_axis, sol = cn.solve(f)

    plt.plot(t_axis, l2(sol, axis=0), label="Numerical (CN)")

    plt.plot(t_axis, [l2(u(t, x_axis)) for t in t_axis], label="Analytical")

    plt.grid()

    plt.suptitle(
        f"Linearized Korteweg-deVries - Conservation of the $l_2$ norm $M={M}, N={N}$"
    )
    plt.xlabel("Time $t$")
    plt.ylabel(r"$l_2$ norm $\|u\|_{l_2}$")

    half_height = 1e-11
    center = np.sqrt(2) / 2
    plt.ylim(ymin=center - half_height, ymax=center + half_height)

    plt.legend()

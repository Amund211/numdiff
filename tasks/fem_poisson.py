from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from numdiff.fem import AFEM
from numdiff.refine import refine_mesh, simple_select_avg, simple_select_max
from numdiff.refinement_utilities import (
    calculate_relative_L2_error_FEM,
    make_FEM_solver,
)


def _task_5_afem_solution(f, u, a, b, u_text, deg=10, tol=1e-3):
    d1 = u(a)
    d2 = u(b)

    N = 20

    alpha_avg = 1.0
    alpha_max = 0.7

    solver_params = {
        "N": N,
        "f": f,
        "u": u,
        "a": a,
        "b": b,
        "d1": d1,
        "d2": d2,
        "tol": tol,
        "deg": deg,
    }

    x_avg, U_avg, ndof_list, error_list = AFEM(
        **solver_params,
        select_refinement=partial(simple_select_avg, alpha=alpha_avg),
    )

    x_max, U_max, ndof_list, error_list = AFEM(
        **solver_params,
        select_refinement=partial(simple_select_max, alpha=alpha_avg),
    )

    fine_x = np.linspace(a, b, max(x_avg.shape[0], x_avg.shape[0], 1000))
    analytical = u(fine_x)

    plt.plot(
        x_avg,
        U_avg,
        label=fr"AFEM avg, $\alpha = {alpha_avg}$, ${x_avg.shape[0]}$ points",
    )
    plt.plot(
        x_max,
        U_max,
        label=fr"AFEM max, $\alpha = {alpha_max}$, ${x_max.shape[0]}$ points",
    )

    plt.plot(fine_x, analytical, linestyle="dashed", label="Analytical solution")

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid()
    plt.legend()

    plt.title(
        fr"$u\left( x \right) = {u_text}, x \in \left[ {a}, {b} \right]$, "
        fr"$tol = {tol:.2e}$ "
    )
    plt.suptitle("Poisson's equation AFEM - averaging vs maximizing")


def _task_5_refinement(f, u, a, b, u_text, scale, order=2, deg=10):
    d1 = u(a)
    d2 = u(b)

    N_array = np.power(2, np.arange(3, 12), dtype=np.int32)

    ndofs, (distances,) = refine_mesh(
        solver=make_FEM_solver(
            a=a,
            b=b,
            f=f,
            d1=d1,
            d2=d2,
            deg=deg,
        ),
        param_range=N_array,
        analytical=u,
        calculate_distances=(calculate_relative_L2_error_FEM,),
    )

    plt.loglog(ndofs, distances, label="$e^r_{L_2}$")

    plt.plot(
        ndofs,
        scale * np.divide(1, (ndofs + 1) ** order),
        linestyle="dashed",
        label=fr"$O\left(h{'^2' if order == 2 else ''}\right)$",
    )

    plt.suptitle("Poisson's equation FEM")
    plt.title(fr"$u\left( x \right) = {u_text}, x \in \left[ {a}, {b} \right] $")
    plt.xlabel("Degrees of freedom $N-1$")
    plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()


def _task_5_afem(
    f,
    u,
    a,
    b,
    u_text,
    tol,
    scale,
    order=2,
    deg=10,
):
    d1 = u(a)
    d2 = u(b)

    N = 20

    alpha_avg = 1.0
    alpha_max = 0.7

    solver_params = {
        "N": N,
        "f": f,
        "u": u,
        "a": a,
        "b": b,
        "d1": d1,
        "d2": d2,
        "tol": tol,
        "deg": deg,
    }

    x_avg, U_avg, ndofs_max, distances_max = AFEM(
        **solver_params,
        select_refinement=partial(simple_select_max, alpha=alpha_max),
    )

    x_avg, U_avg, ndofs_avg, distances_avg = AFEM(
        **solver_params,
        select_refinement=partial(simple_select_avg, alpha=alpha_avg),
    )

    plt.loglog(ndofs_max, distances_max, label=fr"Max, $\alpha = {alpha_max}$")
    plt.loglog(ndofs_avg, distances_avg, label=fr"Avg, $\alpha = {alpha_avg}$")

    log_min_ndof = np.log10(min(ndofs_max.min(), ndofs_avg.min()))
    log_max_ndof = np.log10(max(ndofs_max.max(), ndofs_avg.max()))
    x_axis = np.logspace(log_min_ndof, log_max_ndof, num=1000)

    plt.plot(
        x_axis,
        scale * np.divide(1, (x_axis + 1) ** order),
        linestyle="dashed",
        label=fr"$O\left(h{'' if order == 1 else f'^{{{order}}}'}\right)$",
    )

    plt.suptitle("Poisson's equation AFEM - select max vs avg")
    plt.title(fr"$u\left( x \right) = {u_text}, x \in \left[ {a}, {b} \right] $")
    plt.xlabel("Degrees of freedom $N-1$")
    plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()


b_params = {"f": lambda x: -2, "u": lambda x: x ** 2, "a": 0, "b": 1, "u_text": "x^2"}


def task_5b_refinement():
    _task_5_refinement(**b_params, scale=1e1)


def task_5b_afem():
    _task_5_afem(**b_params, tol=1e-8, scale=1e1)


def task_5b_afem_solution():
    _task_5_afem_solution(**b_params)


c_params = {
    "f": lambda x: -(40000 * x ** 2 - 200) * np.exp(-100 * x ** 2),
    "u": lambda x: np.exp(-100 * x ** 2),
    "a": -1,
    "b": 1,
    "u_text": r"\exp{\left(-100 x^2\right)}",
}


def task_5c_refinement():
    _task_5_refinement(**c_params, scale=2e2)


def task_5c_afem():
    _task_5_afem(**c_params, tol=1e-8, scale=1e1)


def task_5c_afem_solution():
    _task_5_afem_solution(**c_params)


d_params = {
    "f": lambda x: -(4000000 * x ** 2 - 2000) * np.exp(-1000 * x ** 2),
    "u": lambda x: np.exp(-1000 * x ** 2),
    "a": -1,
    "b": 1,
    "u_text": r"\exp{\left(-1000 x^2\right)}",
}


def task_5d_refinement():
    _task_5_refinement(**d_params, scale=2e3)


def task_5d_afem():
    _task_5_afem(**d_params, tol=1e-8, scale=1e1)


def task_5d_afem_solution():
    _task_5_afem_solution(**d_params)


e_params = {
    "f": lambda x: 2 / 9 * x ** (-4 / 3),
    "u": lambda x: x ** (2 / 3),
    "a": 0,
    "b": 1,
    "u_text": r"x^{\frac{2}{3}}",
}


def task_5e_refinement():
    _task_5_refinement(**e_params, scale=1.2e-1, order=1)


def task_5e_afem():
    _task_5_afem(**e_params, tol=4e-8, scale=5e-1, order=1.5)


def task_5e_afem_solution():
    _task_5_afem_solution(**e_params)

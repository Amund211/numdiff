import matplotlib.pyplot as plt
import numpy as np

from conditions import Dirichlet, Neumann
from poisson import poisson
from refine import refine_mesh, select_avg, select_max
from refinement_utilities import (
    calculate_relative_l2_error,
    make_amr_poisson_solver,
    make_calculate_relative_L2_error_poisson,
    make_poisson_solver,
)
from settings import FINE_PARAMETERS


def poisson_1D_UMR(
    f, conditions, analytical, calculate_distances, M_range, plot_kwargs_list
):
    ndofs, distances_list = refine_mesh(
        solver=make_poisson_solver(f=f, conditions=conditions),
        param_range=M_range,
        analytical=analytical,
        calculate_distances=calculate_distances,
    )

    for distances, plot_kwargs in zip(distances_list, plot_kwargs_list):
        plt.loglog(ndofs, distances, **plot_kwargs)


def poisson_1D_AMR(
    f, conditions, analytical, select_refinement, order, M_range, plot_kwargs
):
    ndofs, (distances,) = refine_mesh(
        solver=make_amr_poisson_solver(
            f=f,
            u=analytical,
            conditions=conditions,
            select_refinement=select_refinement,
            order=order,
        ),
        param_range=M_range[M_range != 2],
        analytical=analytical,
        calculate_distances=(make_calculate_relative_L2_error_poisson(f),),
    )

    plt.loglog(ndofs, distances, **plot_kwargs)


def task_1a_solution():
    M = 1000

    alpha = 0
    sigma = 0
    conditions = (
        Dirichlet(condition=alpha, m=0),
        Neumann(condition=sigma, m=-1),
    )

    def f(x):
        return np.cos(2 * np.pi * x) + x

    def u(x):
        # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
        # Here: solved for left dirichlet and right neumann
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (sigma - 1 / 2) * x
            + alpha
        )

    x, U = poisson(f, M, conditions)

    plt.plot(x, U, label="Numerical")

    plt.plot(x, u(x), linestyle="dashed", label="Analytical")

    plt.suptitle(f"Poisson's equation - Analytical vs numerical with $M={M}$")
    plt.title(
        fr"$u_{{xx}} = f(x) = \cos{{\left( 2 \pi x \right)}} + x, u(0, t) = {alpha}, u_x(1, t) = {sigma}$"
    )
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid()
    plt.legend()


def task_1a():
    alpha = 0
    sigma = 0
    conditions = (
        Dirichlet(condition=alpha, m=0),
        Neumann(condition=sigma, m=-1),
    )

    def f(x):
        return np.cos(2 * np.pi * x) + x

    def u(x):
        # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
        # Here: solved for left dirichlet and right neumann
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (sigma - 1 / 2) * x
            + alpha
        )

    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(0, 5.5, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

    poisson_1D_UMR(
        f=f,
        conditions=conditions,
        analytical=u,
        calculate_distances=(
            calculate_relative_l2_error,
            make_calculate_relative_L2_error_poisson(f),
        ),
        M_range=M_range,
        plot_kwargs_list=({"label": "$e^r_{l_2}$"}, {"label": "$e^r_{L_2}$"}),
    )

    plt.plot(
        M_range,
        5 * np.divide(1, (M_range.astype(np.float64) + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.suptitle("Poisson's equation - Uniform mesh refinement")
    plt.title("Dirichlet - Neumann")
    plt.xlabel("Internal nodes $M$")
    plt.ylabel(r"Relative error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()


def task_1b_solution():
    M = 1000

    alpha = 1
    beta = 1
    conditions = (
        Dirichlet(condition=alpha, m=0),
        Dirichlet(condition=beta, m=-1),
    )

    def f(x):
        return np.cos(2 * np.pi * x) + x

    def u(x):
        # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
        # Here: solved for left dirichlet and right dirichlet
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (beta - 7 / 6) * x
            + alpha
        )

    x, U = poisson(f, M, conditions)

    plt.plot(x, U, label="Numerical")

    plt.plot(x, u(x), linestyle="dashed", label="Analytical")

    plt.suptitle(f"Poisson's equation - Analytical vs numerical with $M={M}$")
    plt.title(
        fr"$u_{{xx}} = f(x) = \cos{{\left( 2 \pi x \right)}} + x, u(0, t) = {alpha}, u(1, t) = {beta}$"
    )
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid()
    plt.legend()


def task_1b():
    alpha = 1
    beta = 1
    conditions = (
        Dirichlet(condition=alpha, m=0),
        Dirichlet(condition=beta, m=-1),
    )

    def f(x):
        return np.cos(2 * np.pi * x) + x

    def u(x):
        # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
        # Here: solved for left dirichlet and right dirichlet
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (beta - 7 / 6) * x
            + alpha
        )

    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(0, 5.5, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

    poisson_1D_UMR(
        f=f,
        conditions=conditions,
        analytical=u,
        calculate_distances=(
            calculate_relative_l2_error,
            make_calculate_relative_L2_error_poisson(f),
        ),
        M_range=M_range,
        plot_kwargs_list=({"label": "$e^r_{l_2}$"}, {"label": "$e^r_{L_2}$"}),
    )

    plt.plot(
        M_range,
        10 * np.divide(1, (M_range.astype(np.float64) + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.suptitle("Poisson's equation - Uniform mesh refinement")
    plt.title("Dirichlet - Dirichlet")
    plt.xlabel("Internal nodes $M$")
    plt.ylabel(r"Relative error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()


def task_1c_solution():
    M = 1000

    sigma1 = 0
    sigma2 = 1 / 2
    conditions = (
        Neumann(condition=sigma1, m=0),
        Neumann(condition=sigma2, m=-1),
    )

    def f(x):
        return np.cos(2 * np.pi * x) + x

    def u(x):
        # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
        # Here: solved for left dirichlet and right neumann
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (sigma2 - 1 / 2) * x
            + fake_alpha
        )

    x, U = poisson(f, M, conditions)

    fake_alpha = U[0]  # Used to align the reference analytical solution

    plt.plot(x, U, label="Least squares solution")

    plt.plot(
        x,
        u(x),
        linestyle="dashed",
        label=fr"Analytical ($u(0) \approx {fake_alpha:.4f}$)",
    )

    plt.suptitle(f"Poisson's equation - ill posed problem with $M={M}$")
    plt.title(
        fr"$u_{{xx}} = f(x) = \cos{{\left( 2 \pi x \right)}} + x, u_x(0) = {sigma1}, u_x(1) = {sigma2}$"
    )
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid()
    plt.legend()


def task_1d1():
    # Manufactured solution
    eps = 1 / 1000

    def u(x):
        return np.exp(-1 / eps * np.square(x - 1 / 2))

    def f(x):
        return u(x) * (4 * x ** 2 - 4 * x - 2 * eps + 1) / eps ** 2

    alpha = u(0)
    beta = u(1)

    conditions = (
        Dirichlet(condition=alpha, m=0),
        Dirichlet(condition=beta, m=-1),
    )

    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(0, 4, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

    poisson_1D_AMR(
        f=f,
        conditions=conditions,
        analytical=u,
        select_refinement=select_max,
        order=2,
        M_range=M_range,
        plot_kwargs={"label": "2nd order method"},
    )
    poisson_1D_AMR(
        f=f,
        conditions=conditions,
        analytical=u,
        select_refinement=select_max,
        order=1,
        M_range=M_range,
        plot_kwargs={"label": "1st order method"},
    )

    # Both methods seem to have order 2
    plt.plot(
        M_range,
        1e4 * np.divide(1, (M_range.astype(np.float64) + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.suptitle("Poisson's equation - Adaptive mesh refinement")
    plt.title(
        r"$u\left(x\right) = "
        r"\exp{-\frac{1}{\epsilon} \left(x - \frac12\right)^2},"
        fr"\epsilon = \num{{{eps:.2e}}}$"
    )
    plt.xlabel("Internal nodes $M$")
    plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()


def task_1d2():
    # Manufactured solution
    eps = 1 / 1000

    def u(x):
        return np.exp(-1 / eps * np.square(x - 1 / 2))

    def f(x):
        return u(x) * (4 * x ** 2 - 4 * x - 2 * eps + 1) / eps ** 2

    alpha = u(0)
    beta = u(1)

    conditions = (
        Dirichlet(condition=alpha, m=0),
        Dirichlet(condition=beta, m=-1),
    )

    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(0, 4, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

    poisson_1D_AMR(
        f=f,
        conditions=conditions,
        analytical=u,
        select_refinement=select_max,
        order=2,
        M_range=M_range,
        plot_kwargs={"label": r"AMR max, $\alpha=0.7$"},
    )
    poisson_1D_AMR(
        f=f,
        conditions=conditions,
        analytical=u,
        select_refinement=select_avg,
        order=2,
        M_range=M_range,
        plot_kwargs={"label": r"AMR avg, $\alpha=1$"},
    )
    poisson_1D_UMR(
        f=f,
        conditions=conditions,
        analytical=u,
        calculate_distances=(make_calculate_relative_L2_error_poisson(f),),
        M_range=M_range,
        plot_kwargs_list=({"label": "UMR"},),
    )

    plt.plot(
        M_range,
        1e4 * np.divide(1, (M_range.astype(np.float64) + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.suptitle("Poisson's equation - Adaptive vs uniform mesh refinement")
    plt.title(
        r"$u\left(x\right) = "
        r"\exp{-\frac{1}{\epsilon} \left(x - \frac12\right)^2},"
        fr"\epsilon = \num{{{eps:.2e}}}$"
    )
    plt.xlabel("Internal nodes $M$")
    plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()

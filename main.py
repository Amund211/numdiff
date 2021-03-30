"""
Main file implementing the solutions to the tasks

https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf
https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_2.pdf
"""

import matplotlib.pyplot as plt
import numpy as np

from conditions import Dirichlet, Neumann
from refine import refine_mesh
from refinement_utilities import (
    calculate_relative_l2_error,
    make_calculate_relative_L2_error_poisson,
    make_poisson_solver,
)


def poisson_1D_UMR(
    f, conditions, analytical, calculate_distance, plot_kwargs={"label": r"$\|U-u\|$"}
):
    amt_points, distances = refine_mesh(
        solver=make_poisson_solver(f=f, conditions=conditions),
        param_range=np.unique(np.logspace(0, 3, num=50, dtype=np.int32)),
        analytical=analytical,
        calculate_distance=calculate_distance,
    )

    # Subtract 2 from amt_points bc we have two boundary conditions
    plt.loglog(amt_points - 2, distances, **plot_kwargs)

    plt.legend()
    plt.grid()


if __name__ == "__main__":
    import sys

    plt.rcParams.update({"text.usetex": True})

    available_tasks = ("1a", "1b")

    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        print(f"Available tasks: {' '.join(available_tasks)}")
        tasks = (
            input("What tasks do you want to run? (space separated): ")
            .lower()
            .split(" ")
        )

    for task in tasks:
        if task not in available_tasks:
            print(f"Did not recognize task '{task}', skipping...", file=sys.stdout)
            continue

        if task == "1a":
            alpha = 0
            sigma = 1
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

            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=calculate_relative_l2_error,
                plot_kwargs={"label": r"$\|U-u\|_{l_2}$"},
            )
            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=make_calculate_relative_L2_error_poisson(f),
                plot_kwargs={"label": r"$\|U-u\|_{L_2}$"},
            )

            x = np.logspace(0, 3)
            plt.plot(
                x,
                3 * np.divide(1, x ** 2),
                linestyle="dashed",
                label=r"$O\left(h^2\right)$",
            )

            plt.show()
        elif task == "1b":
            alpha = 0
            beta = 0
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
                    + (beta - 1 / 6) * x
                    + alpha
                )

            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=calculate_relative_l2_error,
                plot_kwargs={"label": r"$\|U-u\|_{l_2}$"},
            )
            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=make_calculate_relative_L2_error_poisson(f),
                plot_kwargs={"label": r"$\|U-u\|_{L_2}$"},
            )

            x = np.logspace(0, 3)
            plt.plot(
                x,
                10 * np.divide(1, x ** 2),
                linestyle="dashed",
                label=r"$O\left(h^2\right)$",
            )
            plt.legend()

            plt.show()
        else:
            raise ValueError(
                f"Task '{task}' present in `available_tasks`, but not implemented"
            )

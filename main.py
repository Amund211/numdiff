"""
Main file implementing the solutions to the tasks

https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf
https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_2.pdf
"""

import matplotlib.pyplot as plt
import numpy as np

from conditions import Dirichlet, Neumann
from plotting import refine_and_plot
from refinement_utilities import calculate_relative_l2_error, make_poisson_solver


def poisson_1D_UMR(conditions, analytical):
    def f(x):
        return np.cos(2 * np.pi * x) + x

    refine_and_plot(
        solver=make_poisson_solver(
            f=f, conditions=conditions, interpolate_result=False
        ),
        analytical=analytical,
        param_range=np.unique(np.logspace(0, 5, num=50, dtype=np.int32)),
        calculate_distance=calculate_relative_l2_error,
    )


if __name__ == "__main__":
    import sys

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
            print(
                f"Did not recognize command '{command}', skipping...", file=sys.stdout
            )
            continue

        if task == "1a":
            alpha = 0
            sigma = 1
            conditions = (
                Dirichlet(condition=alpha, m=0),
                Neumann(condition=sigma, m=-1),
            )

            def u(x):
                # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
                # Here: solved for left dirichlet and right neumann
                return (
                    (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
                    + x ** 3 / 6
                    + (sigma - 1 / 2) * x
                    + alpha
                )

            poisson_1D_UMR(conditions, u)
            plt.show()
        elif task == "1b":
            alpha = 0
            beta = 0
            conditions = (
                Dirichlet(condition=alpha, m=0),
                Dirichlet(condition=beta, m=-1),
            )

            def u(x):
                # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
                # Here: solved for left dirichlet and right dirichlet
                return (
                    (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
                    + x ** 3 / 6
                    + (beta - 1 / 6) * x
                    + alpha
                )

            poisson_1D_UMR(conditions, u)
            plt.show()
        else:
            raise ValueError(
                f"Task '{task}' present in `available_tasks`, but not implemented"
            )

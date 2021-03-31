"""
Main file implementing the solutions to the tasks

https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf
https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_2.pdf
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from conditions import Dirichlet, Neumann
from equations import HeatEquation
from refine import refine_mesh, select_avg, select_max
from refinement_utilities import (
    calculate_relative_l2_error,
    make_amr_poisson_solver,
    make_calculate_relative_L2_error,
    make_calculate_relative_L2_error_poisson,
    make_poisson_solver,
    make_scheme_solver,
)
from schemes import ThetaMethod


def poisson_1D_UMR(f, conditions, analytical, calculate_distance, M_range, plot_kwargs):
    ndofs, distances = refine_mesh(
        solver=make_poisson_solver(f=f, conditions=conditions),
        param_range=M_range,
        analytical=analytical,
        calculate_distance=calculate_distance,
    )

    plt.loglog(ndofs, distances, **plot_kwargs)

    plt.legend()
    plt.grid()


def poisson_1D_AMR(
    f, conditions, analytical, select_refinement, order, M_range, plot_kwargs
):
    ndofs, distances = refine_mesh(
        solver=make_amr_poisson_solver(
            f=f,
            u=analytical,
            conditions=conditions,
            select_refinement=select_refinement,
            order=order,
        ),
        param_range=M_range,
        analytical=analytical,
        calculate_distance=make_calculate_relative_L2_error(),
    )

    plt.loglog(ndofs, distances, **plot_kwargs)

    plt.legend()
    plt.grid()


class HeatTheta(ThetaMethod, HeatEquation):
    def validate_params(self):
        if 1 / 2 <= self.theta <= 1:
            # All r >= 0
            return
        elif 0 <= self.theta < 1 / 2:
            r = self.k / self.h ** 2
            eta = 0
            for condition in self.conditions:
                if isinstance(condition, Neumann):
                    eta = max(eta, abs(condition.condition))

            assert r <= 1 / (
                2 * (1 - 2 * self.theta) * (1 + eta * self.h / 2)
            ), f"r <= 1/(2(1-2theta)(1+eta*h/2)) <= {r} needed for convergence"
        else:
            raise ValueError(f"Invalid value for theta {self.theta}")


if __name__ == "__main__":
    import sys

    # Plot params
    fontsize = 16
    plt.rcParams.update(
        {
            "text.usetex": True,
            "axes.titlesize": fontsize,
            "axes.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "xtick.labelsize": fontsize,
            "lines.linewidth": 2,
            "lines.markersize": 7,
            "legend.fontsize": fontsize,
            "legend.handlelength": 1.5,
            "figure.figsize": (10, 6),
            "figure.titlesize": 20,
        }
    )

    available_tasks = ("1a", "1b", "1d1", "1d2", "2a")

    if len(sys.argv) > 1:
        tasks = sys.argv[1:]
    else:
        print(f"Available tasks: {' '.join(available_tasks)}")
        print("Use 'all' to run all tasks")
        tasks = (
            input("What tasks do you want to run? (space separated): ")
            .lower()
            .split(" ")
        )

    if len(tasks) == 1 and tasks[0] == "all":
        tasks = available_tasks

    for task in tasks:
        if task not in available_tasks:
            print(f"Did not recognize task '{task}', skipping...", file=sys.stdout)
            continue

        print(f"Running task {task}")
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

            M_range = np.unique(np.logspace(0, 3, num=50, dtype=np.int32))

            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=calculate_relative_l2_error,
                M_range=M_range,
                plot_kwargs={"label": "$e^r_{l_2}$"},
            )
            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=make_calculate_relative_L2_error_poisson(f),
                M_range=M_range,
                plot_kwargs={"label": "$e^r_{L_2}$"},
            )

            plt.plot(
                M_range,
                5 * np.divide(1, (M_range + 1) ** 2),
                linestyle="dashed",
                label=r"$O\left(h^2\right)$",
            )

            plt.suptitle("Poisson's equation - Uniform mesh refinement")
            plt.title("Dirichlet - Neumann")
            plt.xlabel("Internal nodes $M$")
            plt.ylabel(r"Relative error $\frac{\|U-u\|}{\|u\|}$")
            plt.legend()

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

            M_range = np.unique(np.logspace(0, 3, num=50, dtype=np.int32))

            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=calculate_relative_l2_error,
                M_range=M_range,
                plot_kwargs={"label": "$e^r_{l_2}$"},
            )
            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=make_calculate_relative_L2_error_poisson(f),
                M_range=M_range,
                plot_kwargs={"label": "$e^r_{L_2}$"},
            )

            x = np.logspace(0, 3)
            plt.plot(
                M_range,
                10 * np.divide(1, (M_range + 1) ** 2),
                linestyle="dashed",
                label=r"$O\left(h^2\right)$",
            )

            plt.suptitle("Poisson's equation - Uniform mesh refinement")
            plt.title("Dirichlet - Dirichlet")
            plt.xlabel("Internal nodes $M$")
            plt.ylabel(r"Relative error $\frac{\|U-u\|}{\|u\|}$")
            plt.legend()

            plt.show()
        elif task == "1d1":
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

            M_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

            poisson_1D_AMR(
                f=f,
                conditions=conditions,
                analytical=u,
                select_refinement=select_max,
                order=2,
                M_range=M_range[M_range != 2],
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
                10000 * np.divide(1, (M_range + 1) ** 2),
                linestyle="dashed",
                label=r"$O\left(h^2\right)$",
            )

            plt.suptitle("Poisson's equation - Adaptive mesh refinement")
            plt.title(
                r"$u\left(x\right) = \exp{-\frac{1}{\epsilon} \left(x - \frac12\right)}$"
            )
            plt.xlabel("Internal nodes $M$")
            plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
            plt.legend()

            plt.show()
        elif task == "1d2":
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

            M_range = np.unique(np.logspace(0, 3, num=30, dtype=np.int32))

            poisson_1D_AMR(
                f=f,
                conditions=conditions,
                analytical=u,
                select_refinement=select_max,
                order=2,
                M_range=M_range[M_range != 2],
                plot_kwargs={"label": r"AMR max, $\alpha=0.7$"},
            )
            poisson_1D_AMR(
                f=f,
                conditions=conditions,
                analytical=u,
                select_refinement=select_avg,
                order=2,
                M_range=M_range[M_range != 2],
                plot_kwargs={"label": r"AMR avg, $\alpha=1$"},
            )
            poisson_1D_UMR(
                f=f,
                conditions=conditions,
                analytical=u,
                calculate_distance=calculate_relative_l2_error,
                M_range=M_range,
                plot_kwargs={"label": "UMR"},
            )

            plt.plot(
                M_range,
                10000 * np.divide(1, (M_range + 1) ** 2),
                linestyle="dashed",
                label=r"$O\left(h^2\right)$",
            )

            plt.suptitle("Poisson's equation")
            plt.xlabel("Internal nodes $M$")
            plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
            plt.legend()

            plt.show()
        elif task == "1d3":
            # Maybe include a concrete example comparing AMR with UMR for some small M?
            pass
        elif task == "2a":
            max_power = 14  # M = 2^max_power - 1 will be used as a reference solution
            T = 0.05
            N = int(1e4)
            theta = 1 / 2

            k = T / N

            def f(x):
                return 2 * np.pi * x + np.sin(2 * np.pi * x)

            scheme_kwargs = {
                "theta": theta,
                "conditions": (Neumann(condition=0, m=0), Neumann(condition=0, m=-1)),
                "N": N,
                "k": k,
            }

            scheme_kwargs_1st_order = scheme_kwargs.copy()
            scheme_kwargs_1st_order["conditions"] = (
                Neumann(condition=0, m=0, order=1),
                Neumann(condition=0, m=-1, order=1),
            )

            # Using M + 2 = 2^p + 1 will make grid points line up
            M_star = 2 ** max_power - 1
            M_range = np.power(2, np.arange(1, max_power)) - 1

            scheme = HeatTheta(
                M=M_star,
                **scheme_kwargs,
            )
            x, solution = scheme.solve(f)

            # Substitute for the analytical solution
            U_star = interp1d(x, solution[:, -1], kind="nearest")

            # 1st order
            ndofs, distances = refine_mesh(
                solver=make_scheme_solver(
                    cls=HeatTheta, f=f, T=T, scheme_kwargs=scheme_kwargs_1st_order
                ),
                param_range=M_range,
                analytical=U_star,
                calculate_distance=calculate_relative_l2_error,
            )
            plt.loglog(ndofs / N, distances, label="1st order")

            # 2nd order
            ndofs, distances = refine_mesh(
                solver=make_scheme_solver(
                    cls=HeatTheta, f=f, T=T, scheme_kwargs=scheme_kwargs
                ),
                param_range=M_range,
                analytical=U_star,
                calculate_distance=calculate_relative_l2_error,
            )
            plt.loglog(ndofs / N, distances, label="2nd order")

            # O(h)
            plt.plot(
                M_range,
                np.divide(1, M_range + 1),
                linestyle="dashed",
                label=r"$O\left(h\right)$",
            )

            # O(h^2)
            plt.plot(
                M_range,
                5 * np.divide(1, (M_range + 1) ** 2),
                linestyle="dashed",
                label=r"$O\left(h^2\right)$",
            )

            plt.grid()

            plt.suptitle(
                "The heat equation - discretization of the boundary conditions"
            )
            plt.title(
                f"Comparison with reference solution with $M^* = {M_star}$ at $t = {T}$"
            )
            plt.xlabel("Internal nodes $M$")
            plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
            plt.legend()

            plt.show()
            pass
        else:
            raise ValueError(
                f"Task '{task}' present in `available_tasks`, but not implemented"
            )

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from fem import AFEM
from refine import refine_mesh, simple_select_avg, simple_select_max
from refinement_utilities import calculate_relative_L2_error_FEM, make_FEM_solver


def _generate_task_5(f, u, a, b, u_text, deg=10, tol=1e-2):
    d1 = u(a)
    d2 = u(b)

    def task_5_refinement():
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
            np.divide(1, (ndofs + 1) ** 2),
            linestyle="dashed",
            label=r"$O\left(h^2\right)$",
        )

        plt.suptitle("Poisson's equation FEM")
        plt.title(fr"$u\left( x \right) = {u_text}, x \in \left[ {a}, {b} \right] $")
        plt.xlabel("Degrees of freedom $N-1$")
        plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
        plt.grid()
        plt.legend()

    def task_5_afem():
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

        x_avg, U_avg = AFEM(
            **solver_params,
            select_refinement=partial(simple_select_avg, alpha=alpha_avg),
        )

        x_max, U_max = AFEM(
            **solver_params,
            select_refinement=partial(simple_select_max, alpha=alpha_avg),
        )

        fine_x = np.linspace(a, b, max(x_avg.shape[0], x_avg.shape[0], 1000))
        analytical = u(fine_x)

        plt.subplot(121)
        plt.plot(fine_x, analytical, linestyle="dashed", label="Analytical solution")
        plt.plot(x_avg, U_avg, label="AFEM solution")
        plt.title(fr"Select avg with $\alpha = {alpha_avg}$, ${x_avg.shape[0]}$ points")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.grid()
        plt.legend()

        plt.subplot(122)
        plt.plot(fine_x, analytical, linestyle="dashed", label="Analytical solution")
        plt.plot(x_max, U_max, label="AFEM solution")
        plt.title(fr"Select max with $\alpha = {alpha_max}$, ${x_max.shape[0]}$ points")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.grid()
        plt.legend()

        plt.suptitle(
            "Poisson's equation AFEM "
            fr"$u\left( x \right) = {u_text}, x \in \left[ {a}, {b} \right]$",
        )

    return (
        task_5_refinement,
        task_5_afem,
    )


task_5b_refinement, task_5b_afem = _generate_task_5(
    f=lambda x: -2, u=lambda x: x ** 2, a=0, b=1, u_text="x^2"
)

task_5c_refinement, task_5c_afem = _generate_task_5(
    f=lambda x: -(40000 * x ** 2 - 200) * np.exp(-100 * x ** 2),
    u=lambda x: np.exp(-100 * x ** 2),
    a=-1,
    b=1,
    u_text=r"\exp{\left(-100 x^2\right)}",
)

task_5d_refinement, task_5d_afem = _generate_task_5(
    f=lambda x: -(4000000 * x ** 2 - 2000) * np.exp(-1000 * x ** 2),
    u=lambda x: np.exp(-1000 * x ** 2),
    a=-1,
    b=1,
    u_text=r"\exp{\left(-1000 x^2\right)}",
)

task_5e_refinement, task_5e_afem = _generate_task_5(
    f=lambda x: 2 / 9 * x ** (-4 / 3),
    u=lambda x: x ** (2 / 3),
    a=0,
    b=1,
    u_text=r"x^{\frac23}",
)

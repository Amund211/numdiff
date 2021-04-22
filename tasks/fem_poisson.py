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
            np.divide(1, (ndofs.astype(np.float64) + 1) ** 2),
            linestyle="dashed",
            label=r"$O\left(h^2\right)$",
        )

        plt.suptitle("Poisson's equation FEM")
        plt.title(fr"$u\left( x \right) = {u_text}, x \in \left[ {a}, {b} \right] $")
        plt.xlabel("Degrees of freedom $N-1$")
        plt.ylabel(r"Relative $L_2$ error $\frac{\|U-u\|}{\|u\|}$")
        plt.grid()
        plt.legend()

    def task_5_AFEM(select_refinement):
        N = 20

        x, U = AFEM(
            N, f, u, a, b, d1, d2, tol=tol, deg=deg, select_refinement=select_refinement
        )

        fine_x = np.linspace(x[0], x[-1], max(x.shape[0], 1000))

        plt.plot(fine_x, u(fine_x), linestyle="dashed", label="Analytical solution")
        plt.plot(x, U, label=f"AFEM solution with ${x.shape[0]}$ points")

        plt.suptitle("Poisson's equation AFEM")
        plt.title(fr"$u\left( x \right) = {u_text}, x \in \left[ {a}, {b} \right]$")
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.grid()
        plt.legend()

    return (
        task_5_refinement,
        partial(task_5_AFEM, simple_select_avg),
        partial(task_5_AFEM, simple_select_max),
    )


task_5b_refinement, task_5b_avg, task_5b_max = _generate_task_5(
    f=lambda x: -2, u=lambda x: x ** 2, a=0, b=1, u_text="x^2"
)

task_5c_refinement, task_5c_avg, task_5c_max = _generate_task_5(
    f=lambda x: -(40000 * x ** 2 - 200) * np.exp(-100 * x ** 2),
    u=lambda x: np.exp(-100 * x ** 2),
    a=-1,
    b=1,
    u_text=r"\exp{\left(-100 x^2\right)}",
)

task_5d_refinement, task_5d_avg, task_5d_max = _generate_task_5(
    f=lambda x: -(4000000 * x ** 2 - 2000) * np.exp(-1000 * x ** 2),
    u=lambda x: np.exp(-1000 * x ** 2),
    a=-1,
    b=1,
    u_text=r"\exp{\left(-1000 x^2\right)}",
)

task_5e_refinement, task_5e_avg, task_5e_max = _generate_task_5(
    f=lambda x: 2 / 9 * x ** (-4 / 3),
    u=lambda x: x ** (2 / 3),
    a=0,
    b=1,
    u_text=r"x^{\frac23}",
)

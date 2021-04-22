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

        X1, U1 = AFEM(
            N, f, u, a, b, d1, d2, tol=tol, deg=deg, select_refinement=select_refinement
        )
        plt.plot(X1, u(X1), ".", color="hotpink")
        plt.plot(X1, U1, color="aqua")
        plt.grid()
        plt.title(len(X1))

    return (
        task_5_refinement,
        partial(task_5_AFEM, simple_select_avg),
        partial(task_5_AFEM, simple_select_max),
    )


task_5b_refinement, task_5b_avg, task_5b_max = _generate_task_5(
    f=lambda x: -2, u=lambda x: x ** 2, a=0, b=1, u_text="x^2"
)

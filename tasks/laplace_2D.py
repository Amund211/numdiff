import matplotlib.pyplot as plt
import numpy as np

from laplace import analytical
from refine import refine_mesh
from refinement_utilities import calculate_relative_l2_error, make_laplace_solver


def task_3bx():
    My = 10 ** 3

    Mx_range = np.unique(np.logspace(0, 2, num=10, dtype=np.int32))
    # Mx_range = np.unique(np.logspace(0, 3, num=100, dtype=np.int32))

    ndofs, (distances,) = refine_mesh(
        solver=make_laplace_solver(
            Mx=None,
            My=My,
        ),
        param_range=Mx_range,
        analytical=analytical,
        calculate_distances=(calculate_relative_l2_error,),
    )

    plt.loglog(ndofs / My, distances, label="$e^r_{l_2}$")

    plt.plot(
        Mx_range,
        2 * np.divide(1, (Mx_range.astype(np.float64) + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.suptitle("Laplace's equation")
    plt.title(f"$x$-refinement with constant $M_y={My}$")
    plt.xlabel("Internal nodes in $x$-direction $M_x$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()

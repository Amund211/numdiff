import matplotlib.pyplot as plt
import numpy as np

from laplace import analytical
from refine import refine_mesh
from refinement_utilities import calculate_relative_l2_error, make_laplace_solver


def task_3bx():
    M = np.array([5, 10, 30, 50, 100])

    ndofs, (distances,) = refine_mesh(
        solver=make_laplace_solver(
            Mx=None,
            My=1000,
        ),
        param_range=M,
        analytical=analytical,
        calculate_distances=(calculate_relative_l2_error,),
    )

    plt.loglog(ndofs, distances, label="$e^r_{l_2}$")

    plt.suptitle("Laplace's equation")
    plt.xlabel("Internal nodes $M_x M_y$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()

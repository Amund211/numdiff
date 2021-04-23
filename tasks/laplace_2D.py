import matplotlib.pyplot as plt
import numpy as np

from laplace import analytical, laplace
from refine import refine_mesh
from refinement_utilities import calculate_relative_l2_error, make_laplace_solver


def task_3_solution():
    M = 100

    (x, y), U = laplace(M, M)
    U = U.reshape((M, M))

    plt.suptitle(f"Laplace's equation - numerical solution with $M_x = M_y = {M}$")

    c = plt.pcolormesh(x, y, U, cmap="hot")
    plt.colorbar(c, ax=plt.gca())
    plt.title(
        r"$u_{xx} + u_{yy} = 0, \> u(x, 1) = \sin{\left( 2 \pi x\right)}, u(0, y) = u(x, 0) = u(1, y) = 0$"
    )
    plt.xlabel("$x$")
    plt.ylabel("$y$")


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


def task_3by():
    Mx = 10 ** 3

    My_range = np.unique(np.logspace(np.log10(2), 2, num=10, dtype=np.int32))
    # My_range = np.unique(np.logspace(np.log10(2), 3, num=100, dtype=np.int32))

    ndofs, (distances,) = refine_mesh(
        solver=make_laplace_solver(
            Mx=Mx,
            My=None,
        ),
        param_range=My_range,
        analytical=analytical,
        calculate_distances=(calculate_relative_l2_error,),
    )

    plt.loglog(ndofs / Mx, distances, label="$e^r_{l_2}$")

    plt.plot(
        My_range,
        2 * np.divide(1, (My_range.astype(np.float64) + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.suptitle("Laplace's equation")
    plt.title(f"$y$-refinement with constant $M_x={Mx}$")
    plt.xlabel("Internal nodes in $y$-direction $M_y$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.grid()
    plt.legend()

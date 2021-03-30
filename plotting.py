import matplotlib.pyplot as plt
import numpy as np

from refine import refine_mesh
from refinement_utilities import calculate_relative_l2_error


def solve_and_plot(scheme, f, analytic=None, transform_x=None):
    """Solve a time evolutions scheme and plot the solutions at different t"""
    x_axis, sol = scheme.solve(f)

    plot_x_axis = transform_x(x_axis) if callable(transform_x) else x_axis

    step_size = max(scheme.N // 10, 1)
    for i in range(0, (scheme.N + 1) // step_size):
        n = i * step_size
        color = f"C{i % 10}"
        plt.plot(
            plot_x_axis, sol[:, n], label=f"U(t={n*scheme.k:.3f}, n={n})", color=color
        )

        if analytic is not None:
            plt.plot(
                plot_x_axis,
                analytic(n * scheme.k, x_axis),
                label=f"u(t={n*scheme.k:.3f})",
                linestyle="dashed",
                color=color,
            )

    plt.legend()
    plt.grid()
    plt.show()

    return sol


def refine_and_plot(
    solver,
    analytical,
    param_range=np.unique(np.logspace(0, 3, num=50, dtype=np.int32)),
    calculate_distance=calculate_relative_l2_error,
):
    """Perform refinement and plot the error in a log-log plot"""
    amt_points, distances = refine_mesh(
        solver=solver,
        param_range=param_range,
        analytical=analytical,
        calculate_distance=calculate_distance,
    )

    plt.loglog(amt_points, distances, label=r"$\|U-u\|$")

    plt.legend()
    plt.grid()
    plt.show()

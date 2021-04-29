from itertools import chain

import matplotlib.pyplot as plt
import numpy as np


def solve_and_plot(scheme, f, analytic=None, transform_x=None):
    """Solve a time evolutions scheme and plot the solutions at different t"""
    x_axis, sol = scheme.solve(f)

    if scheme.periodic:
        x_axis = np.append(x_axis, 1)

    plot_x_axis = transform_x(x_axis) if callable(transform_x) else x_axis

    step_size = max(scheme.N // 10, 1)
    for i in chain(range(0, (scheme.N + 1) // step_size), ((scheme.N) / step_size,)):
        n = int(i * step_size)
        color = f"C{int(i) % 10}"

        if scheme.periodic:
            plt.plot(
                plot_x_axis,
                np.append(sol[:, n], sol[0, n]),
                label=f"U(t={n*scheme.k:.3f}, n={n})",
                color=color,
            )
        else:
            plt.plot(
                plot_x_axis,
                sol[:, n],
                label=f"U(t={n*scheme.k:.3f}, n={n})",
                color=color,
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

    return sol

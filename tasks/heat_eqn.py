import matplotlib.pyplot as plt
import numpy as np
from conditions import Neumann
from equations import HeatEquation
from refine import refine_mesh
from refinement_utilities import calculate_relative_l2_error, make_scheme_solver
from schemes import ThetaMethod
from scipy.interpolate import interp1d


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


def task_2a():
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
    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(
            cls=HeatTheta, f=f, T=T, scheme_kwargs=scheme_kwargs_1st_order
        ),
        param_range=M_range,
        analytical=U_star,
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs / N, distances, label="1st order")

    # 2nd order
    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(cls=HeatTheta, f=f, T=T, scheme_kwargs=scheme_kwargs),
        param_range=M_range,
        analytical=U_star,
        calculate_distances=(calculate_relative_l2_error,),
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

    plt.suptitle("The heat equation - discretization of the boundary conditions")
    plt.title(f"Comparison with reference solution with $M^* = {M_star}$ at $t = {T}$")
    plt.xlabel("Internal nodes $M$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()

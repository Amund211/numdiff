from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from numdiff.conditions import Neumann
from numdiff.equations import HeatEquation
from numdiff.refine import refine_mesh
from numdiff.refinement_utilities import (
    calculate_relative_l2_error,
    make_calculate_relative_L2_error,
    make_scheme_solver,
)
from numdiff.schemes import ThetaMethod
from numdiff.settings import FINE_PARAMETERS


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


def task_2_solution():
    M = 100
    N = 100

    T = 0.2
    theta = 1 / 2

    k = T / N

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    scheme_kwargs = {
        "theta": theta,
        "conditions": (Neumann(condition=0, m=0), Neumann(condition=0, m=-1)),
        "N": N,
        "k": k,
        "M": M,
    }

    scheme = HeatTheta(
        **scheme_kwargs,
    )
    x, solution = scheme.solve(f)

    t, x = np.meshgrid(np.linspace(0, T, N + 1), x)

    c = plt.pcolormesh(x, t, solution, cmap="hot", shading="nearest")
    plt.colorbar(c, ax=plt.gca())

    plt.suptitle(f"The heat equation - numerical solution with $M={M}, N={N}$")
    plt.title(
        r"$u_t = u_{xx}, u(x, 0) = 2 \pi x + \sin{\left( 2 \pi x\right)}, "
        "u_x(0, t) = u_x(1, t) = 0$"
    )
    plt.xlabel("$x$")
    plt.ylabel("Time $t$")


def task_2a():
    # M = 2^max_power - 1 will be used as a reference solution

    if FINE_PARAMETERS:
        N = int(1e5)
        max_power = 16
    else:
        N = int(2e3)
        max_power = 13

    T = 0.05
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
    x, solution = scheme.solve(f, context=1)

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
        np.divide(1, M_range.astype(np.float64) + 1),
        linestyle="dashed",
        label=r"$O\left(h\right)$",
    )

    # O(h^2)
    plt.plot(
        M_range,
        5 * np.divide(1, (M_range.astype(np.float64) + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.grid()

    plt.suptitle("The heat equation - discretization of the boundary conditions")
    plt.title(
        "Comparison with reference solution with "
        f"$M^* = {M_star}$ at $t = {T}$. $N = {N}$"
    )
    plt.xlabel("Internal nodes $M$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()


def _task_2b(default_params, refinement, param_range, get_x_axis):
    T = default_params["T"]
    N = default_params["N"]
    M = default_params["M"]

    def f(x):
        return u(x, 0)

    conditions = (Neumann(condition=0, m=0), Neumann(condition=0, m=-1))

    def u(x, t):
        return np.exp(-t * np.pi ** 2) * np.cos(np.pi * x)

    analytical = partial(u, t=T)

    scheme_kwargs = {
        "conditions": conditions,
        "N": N,
        "M": M,
    }

    # Backward Euler
    ndofs, distances_list = refine_mesh(
        solver=make_scheme_solver(
            cls=HeatTheta,
            f=f,
            T=T,
            **refinement,
            scheme_kwargs={"theta": 1, **scheme_kwargs},
        ),
        param_range=param_range,
        analytical=analytical,
        calculate_distances=(
            calculate_relative_l2_error,
            make_calculate_relative_L2_error(),
        ),
    )
    plt.loglog(
        get_x_axis(ndofs=ndofs, M=M, N=N), distances_list[0], label="BE $e^r_{l_2}$"
    )
    plt.loglog(
        get_x_axis(ndofs=ndofs, M=M, N=N), distances_list[1], label="BE $e^r_{L_2}$"
    )

    # Crank Nicholson
    ndofs, distances_list = refine_mesh(
        solver=make_scheme_solver(
            cls=HeatTheta,
            f=f,
            T=T,
            **refinement,
            scheme_kwargs={"theta": 1 / 2, **scheme_kwargs},
        ),
        param_range=param_range,
        analytical=analytical,
        calculate_distances=(
            calculate_relative_l2_error,
            make_calculate_relative_L2_error(),
        ),
    )
    plt.loglog(
        get_x_axis(ndofs=ndofs, M=M, N=N), distances_list[0], label="CN $e^r_{l_2}$"
    )
    plt.loglog(
        get_x_axis(ndofs=ndofs, M=M, N=N), distances_list[1], label="CN $e^r_{L_2}$"
    )

    # Both ndofs should be the same, because they use the same solver
    return ndofs


# Standard parameters for task 2b
if FINE_PARAMETERS:
    TASK_2B_PARAMS = {
        "T": 0.05,
        "N": 10 ** 4,
        "M": 10 ** 4,
    }
else:
    TASK_2B_PARAMS = {
        "T": 0.05,
        "N": 10 ** 3,
        "M": 10 ** 3,
    }


def task_2bh():
    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(0, 4, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

    _task_2b(
        default_params=TASK_2B_PARAMS,
        refinement={"refine_space": True},
        param_range=M_range,
        get_x_axis=lambda ndofs, M, N: ndofs / N,
    )

    # O(h^2)
    plt.plot(
        M_range,
        9.5e-1 * np.divide(1, (M_range + 1) ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.grid()

    plt.suptitle("The heat equation - $h$-refinement")
    plt.title(
        "Backwards Euler vs Crank Nicholson with "
        f"$N={TASK_2B_PARAMS['N']}$ at $t = {TASK_2B_PARAMS['T']}$"
    )
    plt.xlabel("Internal nodes $M$")
    plt.ylabel(r"Relative error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()


def task_2bk():
    if FINE_PARAMETERS:
        N_range = np.unique(np.logspace(0, 4, num=100, dtype=np.int32))
    else:
        N_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

    _task_2b(
        default_params=TASK_2B_PARAMS,
        refinement={"refine_space": False},
        param_range=N_range,
        get_x_axis=lambda ndofs, M, N: ndofs / M,
    )

    # O(h)
    plt.plot(
        N_range,
        2.2e-1 * np.divide(1, N_range),
        linestyle="dashed",
        label=r"$O\left(h\right)$",
    )

    # O(h^2)
    plt.plot(
        N_range,
        2.2e-2 * np.divide(1, N_range ** 2),
        linestyle="dashed",
        label=r"$O\left(h^2\right)$",
    )

    plt.grid()

    plt.suptitle("The heat equation - $k$-refinement")
    plt.title(
        "Backwards Euler vs Crank Nicholson with "
        f"$M={TASK_2B_PARAMS['M']}$ at $t = {TASK_2B_PARAMS['T']}$"
    )
    plt.xlabel("Time steps $N$")
    plt.ylabel(r"Relative error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()


def task_2bc():
    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(0, 5, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

    c = 1

    ndofs = _task_2b(
        default_params=TASK_2B_PARAMS,
        refinement={"c": c},
        param_range=M_range,
        get_x_axis=lambda ndofs, M, N: ndofs,
    )

    # O(ndofs^(-1))
    plt.plot(
        ndofs,
        1e0 * np.divide(1, ndofs),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-1}\right)$",
    )

    # O(ndofs^(-1/2))
    plt.plot(
        ndofs,
        1e0 * np.divide(1, ndofs ** (1 / 2)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-\frac{1}{2}}\right)$",
    )

    plt.grid()

    plt.suptitle(
        fr"The heat equation - refinement with constant $c=\frac{{k}}{{h}}={c}$"
    )
    plt.title(f"Backwards Euler vs Crank Nicholson at $t = {TASK_2B_PARAMS['T']}$")
    plt.xlabel("Degrees of freedom $N_{dof} = MN$")
    plt.ylabel(r"Relative error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()


def task_2br():
    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(0, 4, num=50, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(0, 3, num=10, dtype=np.int32))

    r = 1

    ndofs = _task_2b(
        default_params=TASK_2B_PARAMS,
        refinement={"r": r},
        param_range=M_range,
        get_x_axis=lambda ndofs, M, N: ndofs,
    )

    # O(ndofs^(-2/3))
    plt.plot(
        ndofs,
        1e0 * np.divide(1, ndofs ** (2 / 3)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-\frac{2}{3}}\right)$",
    )

    plt.grid()

    plt.suptitle(
        fr"The heat equation - refinement with constant $r=\frac{{k}}{{h^2}}={r}$"
    )
    plt.title(f"Backwards Euler vs Crank Nicholson at $t = {TASK_2B_PARAMS['T']}$")
    plt.xlabel("Degrees of freedom $N_{dof} = MN$")
    plt.ylabel(r"Relative error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()

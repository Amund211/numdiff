import matplotlib.pyplot as plt
import numpy as np

from conditions import Dirichlet, Neumann, Periodic
from equations import HeatEquation, InviscidBurgers, InviscidBurgers2, PeriodicKdV
from plotting import solve_and_plot
from poisson import amr, poisson
from refine import refine_mesh
from refinement_utilities import calculate_relative_l2_error, make_scheme_solver
from schemes import RK4, Euler, ThetaMethod


class HeatEuler(Euler, HeatEquation):
    def validate_params(self):
        r = self.k / self.h ** 2
        assert r <= 1 / 2, f"r <= 1/2 <= {r} needed for convergence"


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


class HeatRK4(RK4, HeatEquation):
    pass


class BurgersRK4(RK4, InviscidBurgers):
    pass


class KdVTheta(ThetaMethod, PeriodicKdV):
    pass


def test_poisson():
    def f(x):
        return np.cos(2 * np.pi * x) + x

    # The target for the mesh size
    # The amr method may exceed this number
    M_target = 100
    alpha = 0
    sigma = 1
    beta = 0

    def u(x):
        # 1/(2pi)^2 * (1-cos(2pix)) + 1/6 * x^3 + Ax + B
        # Here: solved for left dirichlet and right dirichlet
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (beta - 1 / 6) * x
            + alpha
        )
        # Here: solved for left dirichlet and right neumann
        return (
            (1 / (2 * np.pi) ** 2) * (1 - np.cos(2 * np.pi * x))
            + x ** 3 / 6
            + (sigma - 1 / 2) * x
            + alpha
        )

    x_adaptive, U_adaptive = amr(
        f=f,
        u=u,
        conditions=(Dirichlet(condition=alpha, m=0), Dirichlet(condition=beta, m=-1)),
        amt_points_target=M_target,
    )
    # plt.plot(x_adaptive, U_adaptive, label="$U_{adaptive}$")

    x_uniform, U_uniform = poisson(
        f=f,
        M=x_adaptive.shape[0],  # Use the same amount of point for a fair comparison
        conditions=(Dirichlet(condition=alpha, m=0), Dirichlet(condition=beta, m=-1)),
        maxiter=1e6,
        explain_solution=True,
    )
    # plt.plot(x_uniform, U_uniform, label="$U_{uniform}$")

    M = 10000
    x = np.arange(0, M + 2).astype(np.float64) * 1 / (M + 1)
    plt.plot(x, u(x), label="u")

    from interpolate import calculate_poisson_derivatives, interpolate

    calculate_derivatives = calculate_poisson_derivatives(f)

    adaptive_interpolated = interpolate(x_adaptive, U_adaptive, calculate_derivatives)
    plt.plot(x, adaptive_interpolated(x), label="$U_{adaptive interpolated}$")

    uniform_interpolated = interpolate(x_uniform, U_uniform, calculate_derivatives)
    plt.plot(x, uniform_interpolated(x), label="$U_{uniform interpolated}$")

    plt.legend()
    plt.show()


def test_heat_euler():
    M = 100
    N = 2000
    k = 1 / (M + 2) ** 2 / 2.5

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    # def f(x):
    # return 2 * x * (x < 1 / 2) + (2 - 2 * x) * (x >= 1 / 2)

    scheme = HeatEuler(
        M=M,
        N=N,
        k=k,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=-1)),
        # conditions=(Neumann(condition=0, m=0), Dirichlet(condition=0, m=-1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=-1,
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=-1,
    )

    solve_and_plot(scheme, f)
    plt.show()


def test_heat_theta():
    M = 1000
    N = 2000
    k = 3.85e-05
    theta = 1 / 2

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    # def f(x):
    # return 2 * x * (x < 1 / 2) + (2 - 2 * x) * (x >= 1 / 2)

    scheme = HeatTheta(
        M=M,
        N=N,
        k=k,
        theta=theta,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=-1)),
        # conditions=(Neumann(condition=0, m=0), Dirichlet(condition=0, m=-1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=M+1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=M+1)),
    )

    solve_and_plot(scheme, f)
    plt.show()


def test_heat_rk4():
    M = 1000
    N = 2000
    N = 200
    k = 3.85e-05

    M = 100
    N = 2000
    k = 1 / (M + 2) ** 2 / 2.5

    def f(x):
        return 2 * np.pi * x + np.sin(2 * np.pi * x)

    scheme = HeatRK4(
        M=M,
        N=N,
        k=k,
        conditions=(Neumann(condition=0, m=0), Neumann(condition=0, m=-1)),
        # conditions=(Neumann(condition=0, m=0), Dirichlet(condition=0, m=-1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=M+1)),
        # conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=2*np.pi, m=M+1)),
    )

    solve_and_plot(scheme, f)
    plt.show()


def test_burgers_rk4():
    # These parameters show breaking at about t = 0.056
    M = 1000
    N = 200000
    k = 1 / (M + 2) ** 2 / 2.5

    # Parameters that run a bit faster
    M = 1000
    N = 2000
    N = 700
    k = 0.0001

    def f(x):
        return np.exp(-400 * (x - 1 / 2) ** 2)

    scheme = BurgersRK4(
        M=M,
        N=N,
        k=k,
        conditions=(Dirichlet(condition=0, m=0), Dirichlet(condition=0, m=-1)),
    )

    solve_and_plot(scheme, f)
    plt.show()


def test_KdV():
    M = 10000
    N = 2000
    N = 20000
    k = 1e-4

    M = 1000
    N = 2000
    k = 1e-3

    def transform_x(x):
        return 2 * (x - 1 / 2)

    def f(x):
        return np.sin(np.pi * transform_x(x))

    def analytic(t, x):
        return np.sin(np.pi * (transform_x(x) - t))

    scheme = KdVTheta(
        M=M - 1,
        N=N,
        k=k,
        conditions=(Periodic(m=0, period=M),),
        theta=1 / 2,  # 1/2 => CN
    )

    solution = solve_and_plot(scheme, f, analytic, transform_x)
    assert np.allclose(solution[0, :], solution[M, :]), "Solution must be periodic"
    plt.show()


def refine_KdV_theta():
    def transform_x(x):
        return 2 * (x - 1 / 2)

    def f(x):
        return np.sin(np.pi * transform_x(x))

    def analytical(t, x):
        return np.sin(np.pi * (transform_x(x) - t))

    scheme_kwargs = {"theta": 1 / 2, "conditions": (Periodic(m=0, period=-1),)}

    T = 1

    from functools import partial

    analytical = partial(analytical, T)

    amt_points, distances = refine_mesh(
        solver=make_scheme_solver(KdVTheta, f=f, T=T, c=1, **scheme_kwargs),
        param_range=np.unique(np.logspace(1, 3, num=50, dtype=np.int32)),
        analytical=analytical,
        calculate_distance=calculate_relative_l2_error,
    )

    # Subtract 2 from amt_points bc we have one boundary condition
    plt.loglog(amt_points - 1, distances, label=r"$\|U-u\|_{l_2}$")

    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    test_poisson()
    test_heat_euler()
    test_heat_theta()
    test_heat_rk4()
    test_burgers_rk4()
    test_KdV()
    refine_KdV_theta()

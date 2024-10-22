from functools import partial
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from numdiff.conditions import FictitiousNeumann
from numdiff.equations import (
    AdvectionDiffusion2ndOrderFictitious,
    PeriodicAdvectionDiffusion1stOrder,
    PeriodicAdvectionDiffusion2ndOrder,
    PeriodicAdvectionDiffusion4thOrder,
)
from numdiff.refine import refine_mesh
from numdiff.refinement_utilities import calculate_relative_l2_error, make_scheme_solver
from numdiff.schemes import ThetaMethod
from numdiff.settings import FINE_PARAMETERS


def task_6_solution():
    M = 100
    N = 200
    T = 0.01
    theta = 1 / 2

    k = T / N

    c = 20
    d = 1

    def f(x):
        return np.sin(4 * np.pi * x)

    def u(x, t):
        return np.exp(-d * (4 * np.pi) ** 2 * t) * np.sin(4 * np.pi * (x + c * t))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (),
        "c": c,
        "d": d,
        "N": N,
        "k": k,
        "M": M,
    }

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion2ndOrder):
        pass

    scheme = Scheme(**scheme_kwargs)
    x, solution = scheme.solve(f)

    t, x = np.meshgrid(np.linspace(0, T, N + 1), np.append(x, 1))

    c = plt.pcolormesh(
        x, t, np.row_stack((solution, solution[0])), cmap="hot", shading="nearest"
    )
    plt.colorbar(c, ax=plt.gca())

    plt.suptitle(
        f"Periodic advection diffusion - numerical solution with $M={M}, N={N}$"
    )
    plt.title(
        r"$u_t = c u_x + d u_{xx}, u(x, 0) = \sin{\left( 4 \pi x\right)}, "
        "u(x, t) = u(x+1, t)$"
    )
    plt.xlabel("$x$")
    plt.ylabel("Time $t$")


def task_6b_refinement():
    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(np.log10(5), 4, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(np.log10(5), 3, num=10, dtype=np.int32))

    theta = 1 / 2
    r = 1

    T = 0.01
    c = 20
    d = 1

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion2ndOrder):
        pass

    def f(x):
        return np.sin(4 * np.pi * x)

    def u(x, t):
        return np.exp(-d * (4 * np.pi) ** 2 * t) * np.sin(4 * np.pi * (x + c * t))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (),
        "c": c,
        "d": d,
    }

    # 2nd order
    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(
            cls=Scheme,
            f=f,
            T=T,
            r=r,
            scheme_kwargs=scheme_kwargs,
        ),
        param_range=M_range,
        analytical=partial(u, t=T),
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs, distances, label="$e^r_{l_2}$")

    # O(Ndof^-2/3))
    plt.plot(
        ndofs,
        6e0 * np.divide(1, ndofs ** (2 / 3)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-\frac{2}{3}}\right)$",
    )

    plt.suptitle(
        r"Periodic advection diffusion - 2nd order space, $\theta = \frac{1}{2}$"
    )
    plt.title(
        fr"Refinement with constant $r=\frac{{k}}{{h^2}}={r}$, $c={c}, d={d}, t={T}$"
    )
    plt.xlabel("$N_{dof}$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()
    plt.grid()


def task_6b_asymptotic():
    if FINE_PARAMETERS:
        M = 1000
    else:
        M = 100

    theta = 1 / 2
    r = 1

    T = 0.5
    c = 20
    d = 1

    N = ceil(T * M ** 2 / r)
    k = T / N

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion2ndOrder):
        pass

    def f(x):
        return np.sin(4 * np.pi * x)

    def u(x, t):
        return np.exp(-d * (4 * np.pi) ** 2 * t) * np.sin(4 * np.pi * (x + c * t))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (),
        "c": c,
        "d": d,
        "M": M,
        "N": N,
        "k": k,
    }

    scheme = Scheme(**scheme_kwargs)
    x, solution = scheme.solve(f, context=1)
    x = np.append(x, 1)

    plt.plot(x, np.append(solution[:, -1], solution[0, -1]), label="Numerical")
    plt.plot(x, u(x, T), label="Analytical")

    plt.suptitle("Periodic advection diffusion")
    plt.title(f"Asymptotic behaviour ($t={T}$) with $M={M}, N={N}$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.grid()


def task_6c():
    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(np.log10(5), 4, num=20, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(np.log10(5), 3, num=10, dtype=np.int32))

    theta = 1 / 2
    r = 1

    T = 0.01
    c = 20
    d = 1

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion2ndOrder):
        pass

    def f(x):
        return np.sin(4 * np.pi * x)

    def u(x, t):
        return np.exp(-d * (4 * np.pi) ** 2 * t) * np.sin(4 * np.pi * (x + c * t))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (),
        "c": c,
        "d": d,
    }

    ndofs, (distances,), runtime = refine_mesh(
        solver=make_scheme_solver(
            cls=Scheme,
            f=f,
            T=T,
            r=r,
            scheme_kwargs=scheme_kwargs,
        ),
        param_range=M_range,
        analytical=partial(u, t=T),
        calculate_distances=(calculate_relative_l2_error,),
        timeit=True,
    )

    plt.suptitle(
        "Periodic advection diffusion - refinement "
        fr"with constant $r=\frac{{k}}{{h^2}}={r}$"
    )

    plt.subplot(121)
    plt.loglog(ndofs, distances, label="$e^r_{l_2}$")

    # O(Ndof^-2/3))
    plt.plot(
        ndofs,
        6e0 * np.divide(1, ndofs ** (2 / 3)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-\frac{2}{3}}\right)$",
    )

    plt.title(f"Error at $t={T}$ (log-log)")
    plt.xlabel("Degrees of freedom $N_{dof}$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()
    plt.grid()

    plt.subplot(122)

    plt.title("Runtime (linear-linear)")

    plt.plot(ndofs, runtime, label="Runtime")

    plt.xlabel("Degrees of freedom $N_{dof}$")
    plt.ylabel("Runtime $t$ $(s)$")
    plt.legend()
    plt.grid()


def task_6d_4th_order():
    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(np.log10(5), 4, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(np.log10(5), 3, num=10, dtype=np.int32))

    theta = 1 / 2
    r = 1

    T = 0.01
    c = 20
    d = 1

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion4thOrder):
        pass

    def f(x):
        return np.sin(4 * np.pi * x)

    def u(x, t):
        return np.exp(-d * (4 * np.pi) ** 2 * t) * np.sin(4 * np.pi * (x + c * t))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (),
        "c": c,
        "d": d,
    }

    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(
            cls=Scheme, f=f, T=T, r=r, scheme_kwargs=scheme_kwargs
        ),
        param_range=M_range,
        analytical=partial(u, t=T),
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs, distances, label="$e^r_{l_2}$")

    # O(ndofs^(-4/3))
    plt.plot(
        ndofs,
        1.5e2 * np.divide(1, ndofs ** (4 / 3)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-\frac{4}{3}}\right)$",
    )

    plt.suptitle("Periodic advection diffusion - 4th order spacial discretization")
    plt.title(
        fr"Refinement with constant $r=\frac{{k}}{{h^2}}={r}$, $c={c}, d={d}, t={T}$"
    )
    plt.xlabel("Degrees of freedom $N_{dof} = MN$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()
    plt.grid()


def task_6d_4th_order_M():
    """
    Currently unused task

    This plots error vs M, with a constant N
    """
    if FINE_PARAMETERS:
        N = 10 ** 5
        M_range = np.unique(np.logspace(np.log10(5), 4, num=100, dtype=np.int32))
    else:
        N = 10 ** 3
        M_range = np.unique(np.logspace(np.log10(5), 3, num=10, dtype=np.int32))

    theta = 1 / 2

    T = 0.01
    c = 20
    d = 1

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion4thOrder):
        pass

    def f(x):
        return np.sin(4 * np.pi * x)

    def u(x, t):
        return np.exp(-d * (4 * np.pi) ** 2 * t) * np.sin(4 * np.pi * (x + c * t))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (),
        "c": c,
        "d": d,
        "N": N,
    }

    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(cls=Scheme, f=f, T=T, scheme_kwargs=scheme_kwargs),
        param_range=M_range,
        analytical=partial(u, t=T),
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs / N, distances, label="$e^r_{l_2}$")

    # O(h^(4))
    plt.plot(
        M_range,
        7e3 * np.divide(1, M_range.astype(np.float64) ** 4),
        linestyle="dashed",
        label=r"$O\left(h^4\right)$",
    )

    plt.suptitle("Periodic advection diffusion - 4th order spacial discretization")
    plt.title(fr"$h$-refinement with $N={N}$")
    plt.xlabel("Internal nodes in $x$-direction $M$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()
    plt.grid()


def task_6d_aperiodic_solution():
    M = 100
    N = 200
    T = 1
    theta = 1 / 2

    k = T / N

    c = 20
    d = 1

    a = (-c + np.sqrt(c ** 2 - 4 * d)) / (2 * d)
    b = (-c - np.sqrt(c ** 2 - 4 * d)) / (2 * d)

    class Scheme(AdvectionDiffusion2ndOrderFictitious, ThetaMethod):
        pass

    def u(x, t):
        return np.exp(-t) * (np.exp(a * x) + np.exp(b * x))

    def f(x):
        return u(x, 0)

    def ux(x, t):
        return np.exp(-t) * (a * np.exp(a * x) + b * np.exp(b * x))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (
            FictitiousNeumann(condition=lambda t: ux(0, t), m=0),
            FictitiousNeumann(condition=lambda t: ux(1, t), m=-1),
        ),
        "c": c,
        "d": d,
        "N": N,
        "k": k,
        "M": M,
    }

    scheme = Scheme(**scheme_kwargs)
    x, solution = scheme.solve(f)

    t, x = np.meshgrid(np.linspace(0, T, N + 1), x)

    c = plt.pcolormesh(x, t, solution, cmap="hot", shading="nearest")
    plt.colorbar(c, ax=plt.gca())

    plt.suptitle(
        f"Aperiodic advection diffusion - numerical solution with $M={M}, N={N}$"
    )
    plt.title(
        "$u_t = c u_x + d u_{xx}, "
        r"u(x, 0) = \exp{\left(ax\right)} + \exp{\left(bx\right)}, "
        "u_x(0, t) = q_1, u_x(1, t) = q_2$"
    )
    plt.xlabel("$x$")
    plt.ylabel("Time $t$")


def task_6d_2nd_order_aperiodic():
    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(np.log10(3), 4, num=100, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(np.log10(3), 3, num=10, dtype=np.int32))

    theta = 1 / 2
    r = 1

    T = 0.01
    c = 20
    d = 1

    a = (-c + np.sqrt(c ** 2 - 4 * d)) / (2 * d)
    b = (-c - np.sqrt(c ** 2 - 4 * d)) / (2 * d)

    class Scheme(AdvectionDiffusion2ndOrderFictitious, ThetaMethod):
        pass

    def u(x, t):
        return np.exp(-t) * (np.exp(a * x) + np.exp(b * x))

    def f(x):
        return u(x, 0)

    def ux(x, t):
        return np.exp(-t) * (a * np.exp(a * x) + b * np.exp(b * x))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (
            FictitiousNeumann(condition=lambda t: ux(0, t), m=0),
            FictitiousNeumann(condition=lambda t: ux(1, t), m=-1),
        ),
        "c": c,
        "d": d,
    }

    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(
            cls=Scheme,
            f=f,
            T=T,
            r=r,
            scheme_kwargs=scheme_kwargs,
        ),
        param_range=M_range,
        analytical=partial(u, t=T),
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs, distances, label="$e^r_{l_2}$")

    # O(Ndof^-2/3))
    plt.plot(
        ndofs,
        1e0 * np.divide(1, ndofs ** (2 / 3)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-\frac{2}{3}}\right)$",
    )

    plt.suptitle("Aperiodic advection diffusion - Neumann - Neumann")
    plt.title(
        fr"Refinement with constant $r=\frac{{k}}{{h^2}}={r}$, $c={c}, d={d}, t={T}$"
    )
    plt.xlabel("Degrees of freedom $N_{dof}$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()
    plt.grid()


def task_6d_1st_order():
    """
    This task is currently unused, as we opted to investigate a method which was first
    order in both time and space. That method ended up being unstable.
    """
    if FINE_PARAMETERS:
        M_range = np.unique(np.logspace(np.log10(3), 5, num=50, dtype=np.int32))
    else:
        M_range = np.unique(np.logspace(np.log10(3), 4, num=10, dtype=np.int32))

    theta = 1 / 2
    c_refinement = 1

    T = 0.01
    c = 20
    d = 1

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion1stOrder):
        pass

    def f(x):
        return np.sin(4 * np.pi * x)

    def u(x, t):
        return np.exp(-d * (4 * np.pi) ** 2 * t) * np.sin(4 * np.pi * (x + c * t))

    scheme_kwargs = {
        "theta": theta,
        "conditions": (),
        "c": c,
        "d": d,
    }

    ndofs, (distances,) = refine_mesh(
        solver=make_scheme_solver(
            cls=Scheme, f=f, T=T, c=c_refinement, scheme_kwargs=scheme_kwargs
        ),
        param_range=M_range,
        analytical=partial(u, t=T),
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs, distances, label="$e^r_{l_2}$")

    # O(ndofs^(-1/2))
    plt.plot(
        ndofs,
        4e0 * np.divide(1, ndofs ** (1 / 2)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-\frac{1}{2}}\right)$",
    )

    plt.suptitle("Periodic advection diffusion - 1st order spacial discretization")
    plt.title(fr"Refinement with constant $c=\frac{{k}}{{h}}={c_refinement}$")
    plt.xlabel("Degrees of freedom $N_{dof} = MN$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()
    plt.grid()

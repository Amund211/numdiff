from functools import partial
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from equations import PeriodicAdvectionDiffusion, PeriodicAdvectionDiffusion4thOrder
from refine import refine_mesh
from refinement_utilities import calculate_relative_l2_error, make_scheme_solver
from schemes import ThetaMethod


def task_6b_refinement():
    M_range = np.unique(np.logspace(np.log10(3), 2, num=10, dtype=np.int32))
    # M_range = np.unique(np.logspace(np.log10(3), 3, num=100, dtype=np.int32))

    theta = 1 / 2
    r = 1

    T = 0.01
    c = 20
    d = 1

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion):
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
            cls=Scheme, f=f, T=T, r=r, scheme_kwargs=scheme_kwargs
        ),
        param_range=M_range,
        analytical=partial(u, t=T),
        calculate_distances=(calculate_relative_l2_error,),
    )
    plt.loglog(ndofs, distances, label="$e^r_{l_2}$")

    # O(ndofs^(-2/3))
    plt.plot(
        ndofs,
        1.5e2 * np.divide(1, ndofs.astype(np.float64) ** (2 / 3)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-\frac23}\right)$",
    )

    plt.suptitle("Periodic advection diffusion")
    plt.title(fr"Refinement with constant $r=\frac{{k}}{{h^2}}={r}$")
    plt.xlabel("Degrees of freedom $N_{dof} = MN$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()
    plt.grid()


def task_6b_asymptotic():
    M = 100
    # M = 1000

    theta = 1 / 2
    r = 1

    T = 0.5
    c = 20
    d = 1

    N = ceil(T * M ** 2 / r)
    k = T / N

    class Scheme(ThetaMethod, PeriodicAdvectionDiffusion):
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
    x, solution = scheme.solve(f)
    x = np.append(x, 1)

    plt.plot(x, np.append(solution[:, -1], solution[0, -1]), label="Numerical")
    plt.plot(x, u(x, T), label="Analytical")

    plt.suptitle("Periodic advection diffusion")
    plt.title(f"Asymptotic behaviour ($T={T}$) with $M={M}, N={N}$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.grid()


def task_6d_4th_order_ndof():
    M_range = np.unique(np.logspace(np.log10(5), 3, num=10, dtype=np.int32))
    # M_range = np.unique(np.logspace(np.log10(5), 3.5, num=100, dtype=np.int32))

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
        1.5e2 * np.divide(1, ndofs.astype(np.float64) ** (4 / 3)),
        linestyle="dashed",
        label=r"$O\left(N_{dof}^{-1}\right)$",
    )

    plt.suptitle("Periodic advection diffusion - 4th order spacial discretization")
    plt.title(fr"Refinement with constant $r=\frac{{k}}{{h^2}}={r}$")
    plt.xlabel("Degrees of freedom $N_{dof} = MN$")
    plt.ylabel(r"Relative $l_2$ error $\frac{\|U-u\|}{\|u\|}$")
    plt.legend()
    plt.grid()


def task_6d_4th_order_M():
    N = 10**3; M_range = np.unique(np.logspace(np.log10(5), 3, num=10, dtype=np.int32))
    # N = 10**5; M_range = np.unique(np.logspace(np.log10(5), 4, num=100, dtype=np.int32))

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
        solver=make_scheme_solver(
            cls=Scheme, f=f, T=T, scheme_kwargs=scheme_kwargs
        ),
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

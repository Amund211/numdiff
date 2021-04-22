import numpy as np
import scipy.sparse.linalg
from scipy.interpolate import interp1d

from integrate import composite, integrate
from refine import refine_after


def phi_up(i, x, x_axis):
    """The first part of the basis function for our function space"""
    return (x - x_axis[i - 1]) / (x_axis[i] - x_axis[i - 1])


def phi_down(i, x, x_axis):
    """The second part of the basis function for our function space"""
    return (x_axis[i + 1] - x) / (x_axis[i + 1] - x_axis[i])


def FEM(x_axis, g, d1, d2, deg):
    """Solve poissons equation with the FEM the grid `x_axis`"""
    amt_points = x_axis.shape[0]
    N = amt_points - 1
    h_inv = 1 / (x_axis[1:] - x_axis[:-1])

    main_diag = np.append(h_inv, 0) + np.append(0, h_inv)

    A = scipy.sparse.diags(
        (-h_inv, main_diag, -h_inv),
        (-1, 0, 1),
        shape=(amt_points, amt_points),
        format="lil",
        dtype=np.float64,
    )

    f = np.zeros(amt_points)

    for i in range(1, N - 1):
        f[i] = integrate(
            lambda x: phi_up(i, x, x_axis) * g(x), x_axis[i - 1], x_axis[i], deg=deg
        ) + integrate(
            lambda x: phi_down(i, x, x_axis) * g(x), x_axis[i], x_axis[i + 1], deg=deg
        )

    u = np.zeros(amt_points)
    u[0] = d1
    u[-1] = d2

    f = f - A.tocsr().dot(u)

    A = A[1:-1, 1:-1]
    f = f[1:-1]
    u_bar = scipy.sparse.linalg.spsolve(A.tocsc(), f)
    u[1:-1] = u_bar

    return x_axis, u


def FEM_uniform(N, a, b, g, d1, d2, deg=10):
    """Solve poissons equation with the FEM on a uniform grid of N intervals"""
    x = np.linspace(a, b, N + 1)
    return FEM(x, g, d1, d2, deg=deg)


def AFEM(N, f, u, a, b, d1, d2, tol, deg, select_refinement):
    """Adaptively refine the mesh until the global error is < tol"""
    x = np.linspace(a, b, N + 1, dtype=np.float64)
    to_refine = np.array((), dtype=np.int32)
    global_error = float("inf")

    while global_error >= tol:
        x = refine_after(x, to_refine)
        x, U = FEM(x, f, d1, d2, deg=deg)

        # Linear interpolation of the weights is the same as the weighted sum
        # of the basis functions
        interpolated = interp1d(x, U, kind="linear")
        # The square error on each interval
        err = integrate(lambda x: (u(x) - interpolated(x)) ** 2, x[:-1], x[1:])

        to_refine = select_refinement(np.sqrt(err))

        # e^r_{L_2}
        global_error = np.sqrt(np.sum(err) / composite(lambda x: u(x) ** 2, x))

    return x, U

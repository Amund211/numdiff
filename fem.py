import numpy as np
import scipy.sparse.linalg
from scipy.interpolate import CubicSpline

from integrate import integrate


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


def c_norm(x, y, a, b):
    """The continuous L_2-norm for a function (represented as a vector y)"""
    X, Y = np.polynomial.legendre.leggauss(10)
    # shifting X:
    for i in range(10):
        X[i] = (b - a) / 2 * X[i] + (b + a) / 2
    U = (CubicSpline(x, y)(X)) ** 2
    I = (b - a) / 2 * (sum(Y[j] * U[j] for j in range(10)))
    return np.sqrt(I)


def error_norm(u, Uc, a, b):
    X = np.linspace(a, b, 10)
    I = (u(X) - Uc(X)) ** 2
    x, Y = np.polynomial.legendre.leggauss(10)
    # shifting X:
    for i in range(10):
        x[i] = (b - a) / 2 * x[i] + (b + a) / 2
    I = (b - a) / 2 * (sum(Y[j] * I[j] for j in range(10)))
    return np.sqrt(I)


def AFEM(N, f, u, a, b, d1, d2, alpha, estimate="averaging", deg=10):
    X, U = FEM_uniform(N, a, b, f, d1, d2, deg=deg)
    # plt.plot(X,U,color='green')
    errors = 10 * np.ones(len(X) - 1)

    for j in range(5):
        if estimate == "averaging":
            """Averaging error estimate:"""
            E = alpha * c_norm(X, u(X) - U, a, b) / N
        if estimate == "maximum":
            maximum = (np.absolute(u(X) - U)).max()
            E = alpha * maximum

        Uc = CubicSpline(X, U)

        """Error on each grid-interval:"""
        for i in range(len(X) - 1):
            errors[i] = error_norm(u, Uc, X[i], X[i + 1])

        if np.sum(errors) < len(errors) * E:
            return X, U

        """Adding grid-points where it is necessary:"""
        k = 0
        for i, e in enumerate(errors):
            if e >= E:
                X = np.insert(X, i + 1 + k, (X[i + 1 + k] + X[i + k]) / 2)
                k += 1
        """Making sure that the first two grid elements are of same length"""
        if np.abs((X[1] - X[0]) - (X[2] - X[1])) >= 1e-5:
            X = np.insert(X, 1, (X[1] + X[0]) / 2)

        """Solving the system with respect to the new grid"""
        X, U = FEM(X, f, d1, d2, deg=deg)
        errors = np.ones(len(X) - 1)
    return X, U

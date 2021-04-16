import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg
from scipy import linalg
from scipy.interpolate import CubicSpline

from integrate import integrate


def d_norm(x):
    """The discerete l_2-norm for a vector x"""
    l = 1 / np.sqrt(len(x))
    return l * linalg.norm(x)


def c_norm(x, y, a, b):
    """The continuous L_2-norm for a function (represented as a vector y)"""
    X, Y = np.polynomial.legendre.leggauss(10)
    # shifting X:
    for i in range(10):
        X[i] = (b - a) / 2 * X[i] + (b + a) / 2
    U = (CubicSpline(x, y)(X)) ** 2
    I = (b - a) / 2 * (sum(Y[j] * U[j] for j in range(10)))
    return np.sqrt(I)


def phi_up(i, x, x_axis):
    """The first part of the basis function for our function space"""
    return (x - x_axis[i - 1]) / (x_axis[i] - x_axis[i - 1])


def phi_down(i, x, x_axis):
    """The second part of the basis function for our function space"""
    return (x_axis[i + 1] - x) / (x_axis[i + 1] - x_axis[i])


def FEM(N, a, b, g, d1, d2, deg=10):
    amt_points = N + 1
    h = (b - a) / N
    x_axis = np.linspace(a, b, amt_points)

    A = scipy.sparse.diags(
        (-1 / h, 2 / h, -1 / h),
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


deg = 20


def FEM_error_plot(N_array, a, b, f, d0, d1, u):
    error = np.zeros(len(N_array))
    for i, N in enumerate(N_array):
        x, U = FEM(N, a, b, f, d0, d1, deg)
        error[i] = c_norm(x, u(x) - U, a, b) / (c_norm(x, u(x), a, b))
    plt.plot(N_array, error)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()


# In[ ]:


def error_norm(u, Uc, a, b):
    X = np.linspace(a, b, 10)
    I = (u(X) - Uc(X)) ** 2
    x, Y = np.polynomial.legendre.leggauss(10)
    # shifting X:
    for i in range(10):
        x[i] = (b - a) / 2 * x[i] + (b + a) / 2
    I = (b - a) / 2 * (sum(Y[j] * I[j] for j in range(10)))
    return np.sqrt(I)


def AFEM(N, f, u, a, b, d1, d2, alpha, estimate="averaging"):
    X = np.linspace(a, b, N)
    X, U = FEM(X, a, b, f, d1, d2, deg=10)
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
        X, U = FEM_non_uniform(X, f, a, b, d1, d2)
        errors = np.ones(len(X) - 1)
    return X, U


def FEM_non_uniform(X, g, alpha, beta, d1, d2):
    N = len(X)
    h = np.zeros(N - 1)
    h2 = np.zeros(N)
    """The grid elements:"""
    for i in range(0, N - 1):
        h[i] = 1 / (X[i + 1] - X[i])
    for i in range(1, N - 1):
        h2[i] = h[i] + h[i - 1]
    h2[0] = h[0]
    h2[-1] = h[-1]
    A = np.diag(h2) + np.diag(-h, k=-1) + np.diag(-h, k=1)
    f = np.zeros(N)
    x, y = np.polynomial.legendre.leggauss(deg)
    for i in range(N - 1):
        ai = X[i]
        bi = X[i + 1]
        f[i] = (
            (bi - ai)
            / 2
            * (sum(y[j] * g((bi - ai) / 2 * x[j] + (ai + bi) / 2) for j in range(deg)))
        )
    u = np.zeros(N)
    u[0] = d1
    u[-1] = d2
    f = f - np.dot(A, u)
    A = A[1:-1, 1:-1]
    A = csc_matrix(A)
    f = f[1:-1]
    U = spsolve(A, f)
    U = np.append(np.array(d1), U)
    U = np.append(U, d2)
    # plt.plot(X,U,color='aqua')
    return X, U

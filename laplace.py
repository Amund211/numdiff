import numpy as np
from scipy import linalg


def d_norm(x):
    l = 1 / np.sqrt(len(x))
    return l * linalg.norm(x)


def analytical(Mx, My):
    h = 1 / (Mx + 1)
    k = 1 / (My + 1)
    u = np.zeros(Mx * My)
    j = 1
    l = 1
    X = np.zeros((Mx * My, 2))
    for i in range(Mx * My):
        x = l * h
        y = j * k
        X[i] = [x, y]
        u[i] = (
            (1 / (np.sinh(2 * np.pi))) * np.sin(2 * np.pi * x) * np.sinh(2 * np.pi * y)
        )
        if j % My == 0:
            j = 0
            l += 1
        j += 1
    return u, X


def Laplace(Mx, My):
    M = Mx * My
    f = np.zeros(Mx * My)
    h = 1 / (Mx + 1)
    k = 1 / (My + 1)
    A = (
        np.diag(-4 * np.ones(M))
        + np.diag(np.ones(M - 1), k=1)
        + np.diag(np.ones(M - My), k=My)
        + np.diag(np.ones(M - My), k=-My)
        + np.diag(np.ones(M - 1), k=-1)
    )
    i = 1
    for j in range(M + 1):
        if j % My == 0 and j < M:
            A[j - 1, j] = 0
            A[j, j - 1] = 0
        if j % My == 0 and j != 0:
            f[j - 1] = np.sin(2 * np.pi * i * h)
            i += 1
    A = -A
    U = linalg.solve(A, f)

    return U

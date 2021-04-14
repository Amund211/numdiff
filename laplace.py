"""
This module considers the following 2D Laplace equation on the unit square (x,y) ∈ [0,1]^2 = Ω
u_xx + u_yy = 0
u(0,y) = 0, 0≤y≤1
u(x,0) = 0, 0≤x≤1
u(1,y) = 0, 0≤y≤1
u(x,1) = sin(2πx), 0≤x≤1.
"""

import numpy as np
import scipy.sparse.linalg


def analytical(X, Y):
    """
    Calculate the analytical solution as returned by `laplace`

    X and Y should be a meshgrid with indexing ij
    """
    meshed_result = (
        (1 / (np.sinh(2 * np.pi))) * np.sin(2 * np.pi * X) * np.sinh(2 * np.pi * Y)
    )
    return meshed_result.reshape(-1)


def laplace(Mx, My):
    M = Mx * My
    f = np.zeros(Mx * My)
    h = 1 / (Mx + 1)
    k = 1 / (My + 1)
    A = scipy.sparse.diags(
        (-1, -1, 4, -1, -1),
        (-My, -1, 0, 1, My),
        shape=(M, M),
        format="lil",
        dtype=np.float64,
    )
    i = 1
    for j in range(M + 1):
        if j % My == 0 and j < M:
            A[j - 1, j] = 0
            A[j, j - 1] = 0
        if j % My == 0 and j != 0:
            f[j - 1] = np.sin(2 * np.pi * i * h)
            i += 1
    U = scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), f)

    X, Y = np.meshgrid(
        h * np.arange(1, Mx + 1), k * np.arange(1, My + 1), indexing="ij"
    )

    return (X, Y), U

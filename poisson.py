# https://wiki.math.ntnu.no/_media/tma4212/2021v/tma4212_project_1.pdf
import numpy as np

def central_difference(N, order=2):
    diag = -2 * np.ones(N)
    offdiag = np.ones(N - 1)

    return np.diag(diag) + np.diag(offdiag, k=-1) + np.diag(offdiag, k=1)


def poisson(f, M, alpha, sigma):
    assert M >= 4

    h = 1/(M+1)
    A = central_difference(M+1, order=2) / h**2

    # x1=h, x2=2h, ..., xm+1 = 1
    x = np.arange(1, M + 2) * h
    f = f(x)


    # Dirichlet
    f[0] -= alpha / h**2

    # Adjust for neumann in right endpoint
    #A[-1, -3:] = (-1/(2*h), 2/h, - 3/(2*h))
    # Provided schema has wrong signs
    A[-1, -3:] = (1/(2*h), -2/h, 3/(2*h))
    f[-1] = sigma

    U = np.linalg.solve(A, f)

    return x, np.linalg.solve(A, f)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def f(x):
        return np.cos(2 * np.pi * x) + x

    M = 9
    alpha = 0
    sigma = 1

    def u(x):
        return (1 / (2 * np.pi)**2) * (1-np.cos(2*np.pi*x)) + x**3/6 + (sigma - 1/2)*x + alpha

    x, U = poisson(f, M, alpha, sigma)

    plt.plot(x, U, label="U")
    plt.plot(x, u(x), label="u")
    plt.legend()
    plt.show()



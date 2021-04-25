from numpy.polynomial.legendre import leggauss

from cache import cache


@cache
def get_gauss_nodes(deg):
    """Return nodes and weights for Gauss-Legendre quadrature"""
    nodes, weights = leggauss(deg)
    return nodes, weights


def integrate(f, a, b, deg=10):
    """Compute the integral of f from a to b using Gauss-Legendre quadrature"""
    nodes, weights = get_gauss_nodes(deg)

    scale = (b - a) / (1 - (-1))
    shift = scale + a

    def transform(x):
        """Affine transform from [-1, 1] to [a, b]"""
        return x * scale + shift

    return scale * sum(w * f(transform(x)) for x, w in zip(nodes, weights))


def composite(f, x, deg=10):
    """
    Compute the integral of f on the grid `x` using composite Gauss-Legendre quadrature
    """
    return sum(integrate(f, x[i], x[i + 1]) for i in range(x.shape[0] - 1))

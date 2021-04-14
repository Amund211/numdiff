from helpers import relative_l2_error, relative_L2_error
from interpolate import calculate_poisson_derivatives, interpolate
from laplace import laplace
from poisson import amr, poisson


def calculate_relative_l2_error(x, analytical, numerical):
    """Helper to calculate discrete e^r_l2"""
    return relative_l2_error(analytical(x), numerical)


def make_calculate_relative_L2_error(bc_type=None):
    def calculate_relative_L2_error(x, analytical, numerical):
        """Helper to calculate continuous e^r_L2"""
        kwargs = {"bc_type": bc_type} if bc_type is not None else {}
        interpolated = interpolate(x, numerical, **kwargs)
        return relative_L2_error(analytical, interpolated, x)

    return calculate_relative_L2_error


def make_calculate_relative_L2_error_poisson(f):
    def calculate_relative_L2_error_poisson(x, analytical, numerical):
        """Helper to calculate continuous e^r_L2"""
        interpolated = interpolate(x, numerical, calculate_poisson_derivatives(f))
        return relative_L2_error(analytical, interpolated, x)

    return calculate_relative_L2_error_poisson


def make_poisson_solver(f, conditions):
    """
    Create a solver function for solving Poisson's equation for use in `refine_mesh`
    """

    def solver(param):
        x, U = poisson(f=f, conditions=conditions, M=param)
        return x, U, param

    return solver


def make_amr_poisson_solver(f, u, conditions, select_refinement, order):
    """
    Create a solver function for solving Poisson's equation for use in `refine_mesh`
    """

    def solver(param):
        x, U = amr(
            f=f,
            u=u,
            conditions=conditions,
            M=param,
            select_refinement=select_refinement,
            order=order,
        )
        return x, U, param

    return solver


def make_scheme_solver(cls, f, T, refine_space=True, r=None, c=None, scheme_kwargs={}):
    """
    Create a solver function from a time evolution scheme for use in `refine_mesh`

    f: the function passed to scheme.solve
    T: the time to find the solution for
    refine_space:
        if True, the parameter given to the returned solver function is interpreted
        as M. Otherwise it is interpreted as N
    r: if not None keep a constant r = k/h^2 during refinement
    c: if not None keep a constant c = k/h during refinement
    """
    assert not (
        r is not None and c is not None
    ), "May only keep one of r and c constant at a time"
    assert refine_space or (
        r is None and c is None
    ), "Constant r or k is only supported when `refine_space` is True"

    def solver(param):
        if refine_space:
            M = scheme_kwargs["M"] = param
            if r is not None:
                # Keep a constant r = k/h^2
                scheme_kwargs["N"] = int((M + 1) ** 2 / r)
            elif c is not None:
                # Keep a constant c = k/h
                scheme_kwargs["N"] = int((M + 1) / c)
        else:
            scheme_kwargs["N"] = param

        scheme_kwargs["k"] = T / scheme_kwargs["N"]

        scheme = cls(**scheme_kwargs)
        x_axis, solution = scheme.solve(f)
        return (
            x_axis,
            solution[:, -1],
            # ndof = (#degrees of freedom in x) * (#degrees of freedom in t)
            scheme.free_indicies.shape[0] * scheme.N,
        )

    return solver


def make_laplace_solver(Mx=None, My=None):
    """
    Create a solver function for solving Laplace's equation for use in `refine_mesh`
    """

    def solver(param):
        M = (localMx := Mx or param) * (localMy := My or param)
        meshgrid, U = laplace(Mx=localMx, My=localMy)
        return meshgrid, U, M

    return solver

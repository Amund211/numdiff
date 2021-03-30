from helpers import relative_l2_error, relative_L2_error
from interpolate import calculate_poisson_derivatives, interpolate
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
        return x, U

    return solver


def make_amr_poisson_solver(f, u, conditions, select_refinement):
    """
    Create a solver function for solving Poisson's equation for use in `refine_mesh`
    """

    def solver(param):
        x, U = amr(
            f=f,
            u=u,
            conditions=conditions,
            amt_points_target=param,
            select_refinement=select_refinement,
        )
        return x, U

    return solver


def make_scheme_solver(cls, f, T, refine_space=True, r=None, c=None, **kwargs):
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
            M = kwargs["M"] = param
            if r is not None:
                # Keep a constant r = k/h^2
                kwargs["N"] = int((M + 1) ** 2 / r)
            elif c is not None:
                # Keep a constant c = k/h
                kwargs["N"] = int((M + 1) / c)
        else:
            kwargs["N"] = param

        kwargs["k"] = T / kwargs["N"]

        scheme = cls(**kwargs)
        x_axis, solution = scheme.solve(f)
        return x_axis, solution[:, -1]

    return solver

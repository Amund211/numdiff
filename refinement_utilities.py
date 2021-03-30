from helpers import relative_l2_error, relative_L2_error
from interpolate import interpolate


def calculate_relative_l2_error(x, analytical, numerical):
    """Helper to calculate discrete e^r_l2"""
    return relative_l2_error(analytical(x), numerical)


def calculate_relative_L2_error(x, analytical, numerical):
    """Helper to calculate continuous e^r_L2"""
    return relative_L2_error(analytical, numerical, x)


def make_scheme_solver(
    cls,
    f,
    T,
    refine_space=True,
    r=None,
    c=None,
    interpolate_result=False,
    bc_type="not-a-knot",
    **kwargs
):
    """
    Create a solver function from a time evolution scheme for use in `refine_mesh`

    f: the function passed to scheme.solve
    T: the time to find the solution for
    refine_space:
        if True, the parameter given to the returned solver function is interpreted
        as M. Otherwise it is interpreted as N
    r: if not None keep a constant r = k/h^2 during refinement
    c: if not None keep a constant c = k/h during refinement
    interpolate_result: whether to interpolate the solution. Set to True when using L2
    bc_type: bc_type passed to CubicSpline when interpolate_result is True
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
        if interpolate_result:
            return x_axis, interpolate(x_axis, solution[:, -1], bc_type=bc_type)
        else:
            return x_axis, solution[:, -1]

    return solver

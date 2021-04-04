from functools import cache

import numpy as np
import scipy.sparse.linalg
from scipy.sparse import eye as sparse_eye


class Scheme:
    """
    A general scheme class for linear 1D time evolution equations u_t = Lu

    Specify L by inheriting from an `Equation`
    Specify the time discretization in `.matrix()` and `.rhs()`.

    If the time discretization is explicit, the operator need not be linear.

    Uses a direct method for solving the linear system, override `.get_solver()` to use
    an iterative method. (ex.: scipy.sparse.linalg.cg)
    """

    def __init__(self, *, k, N, factorize=True, **kwargs):
        self.k = k
        self.N = N
        # Whether the step matrix should be factorized, set to False if it often changes
        self.factorize = factorize

        super().__init__(**kwargs)

        assert self.k > 0
        assert self.N > 0
        self.validate_params()

    def validate_params(self):
        """Validate that the method is convergent with the given parameters"""

        pass

    def matrix(self):
        """
        A matrix representing the left hand side (t_n+1) of the discretized system
        in time.
        """

        raise NotImplementedError

    def rhs(self, context, n):
        """
        The right hand side (t_n) corresponding to the time discretized system given
        by `.matrix()`
        """

        raise NotImplementedError

    def get_constrained_rhs(self, context, n):
        b = self.rhs(context, n)
        for condition, index in zip(self.conditions, self.restricted_indicies):
            scalar = condition.get_scalar(n * self.k)
            b[index] = scalar

        return b

    @cache
    def get_constrained_matrix(self):
        assert len(self.conditions) == len(
            self.restricted_indicies
        ), f"Must have exactly  {len(self.restricted_indicies)} boundary conditions"

        # Convert to lil before setting the boundary condition rows
        A = self.matrix().tolil()

        for condition, index in zip(self.conditions, self.restricted_indicies):
            eqn = condition.get_vector(length=self.length, h=self.h)
            A[index, :] = eqn

        return A.tocsc()

    @cache
    def get_solver(self):
        sparse = self.get_constrained_matrix()
        if self.factorize:
            return scipy.sparse.linalg.factorized(sparse)
        else:
            return lambda b: scipy.sparse.linalg.spsolve(sparse, b)

    def step(self, context, n):
        return self.get_solver()(self.get_constrained_rhs(context, n))

    def solve(self, f):
        """
        Solve the time evolution equation
        """

        sol = np.empty((self.length, self.N + 1), dtype=np.float64)
        x_axis = np.linspace(0, 1, self.length)
        sol[:, 0] = f(x_axis)

        for n in range(1, self.N + 1):
            U = self.step(sol, n)

            sol[:, n] = U

        return x_axis, sol


class Euler(Scheme):
    def rhs(self, context, n):
        rhs = np.empty((self.length,), dtype=np.float64)
        rhs[self.restricted_indicies] = 0

        rhs[self.free_indicies] = context[
            self.free_indicies, n - 1
        ] + self.k * self.apply_operator(n, context[:, n - 1])
        return rhs

    def matrix(self):
        # Euler is explicit, so no need to solve a system => identity
        return sparse_eye(self.length, format="lil")


class ThetaMethod(Scheme):
    """
    The theta method
    theta=0 => euler
    theta=1/2 => CN
    theta=1 => implicit euler
    """

    def __init__(self, *, theta, **kwargs):
        self.theta = theta
        super().__init__(**kwargs)

    def rhs(self, context, n):
        rhs = np.empty((self.length,), dtype=np.float64)
        rhs[self.restricted_indicies] = 0

        rhs[self.free_indicies] = context[self.free_indicies, n - 1] + (
            1 - self.theta
        ) * self.k * self.apply_operator(n, context[:, n - 1])

        return rhs

    def matrix(self):
        # csc has fast matrix addition
        return (
            sparse_eye(self.length, format="csc")
            - self.theta * self.k * self.operator().tocsc()
        )


class RK4(Scheme):
    def rhs(self, context, n):
        rhs = np.empty((self.length,), dtype=np.float64)
        rhs[self.restricted_indicies] = 0

        k_1 = self.apply_operator(n, context[:, n - 1], restrict=False)
        k_2 = self.apply_operator(
            n + self.k / 2,
            context[:, n - 1] + self.k / 2 * k_1,
            restrict=False,
        )
        k_3 = self.apply_operator(
            n + self.k / 2,
            context[:, n - 1] + self.k / 2 * k_2,
            restrict=False,
        )
        k_4 = self.apply_operator(
            n + self.k, context[:, n - 1] + self.k * k_3, restrict=False
        )

        rhs[self.free_indicies] = context[
            self.free_indicies, n - 1
        ] + self.k / 6 * self.restrict(k_1 + 2 * k_2 + 2 * k_3 + k_4)

        return rhs

    def matrix(self):
        return sparse_eye(self.length, format="lil")

import numpy as np
from scipy import linalg

from .helpers import pretty_str


def print_sigmas(n=1, mean=5, cov=3, κ=2, α=0.1, β=2.0):
    """_summary_

    Parameters
    ----------
    n : int, optional
        _description_, by default 1
    mean : int, optional
        _description_, by default 5
    cov : int, optional
        _description_, by default 3
    """

    points = MerweScaledSigmas(n, κ, α, β)
    print(f"sigmas: {points.sigma_points(mean, cov).T[0]}")
    Wm, Wc = points.Wm, points.Wc
    print(f"mean weights: {Wm}")
    print(f"cov weights: {Wc}")
    print(f"lambda: {α**2 * (n + κ) - n}")
    print(f"sum cov: {sum(Wc)}")


class JulierSigmas:
    """
    Generates sigma points and weights according to Simon J. Julier
    and Jeffery K. Uhlmann's original paper.
    It parametizes the sigma points using κ.
    """

    def __init__(self, n, kappa=0.0, sqrt_method=None, subtract=None):
        self.n = n
        self.κ = kappa
        self.sqrt = linalg.cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        self.sigma_weights()

    def num_sigmas(self):
        """Number of sigma points for each variable in the state x"""
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        """Computes the sigma points for an UKF given the mean (x) and covariance (P) of the filter.
        κ is an arbitrary constant. Returns sigma points.
        """

        if self.n != np.size(x):
            raise ValueError(f"expected size(x) {self.n}, but size is {np.size(x)}")

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        n = np.size(x)
        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)
        sigmas = np.zeros((2 * n + 1, n))

        # U'*U = (n+κ)*P.
        # Returns lower triangular matrix.
        U = self.sqrt((n + self.κ) * P)

        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x, U[k])
        return sigmas

    def sigma_weights(self):
        """Computes the weights for the UKF. In this formulation the weights for the mean and covariance are the same."""

        n = self.n
        k = self.κ
        c = n + k

        self.Wm = np.full(2 * n + 1, 0.5 / c)
        self.Wm[0] = k / c
        self.Wc = self.Wm

    def __repr__(self):
        return "\n".join(
            [
                "JulierSigmas object",
                pretty_str("n", self.n),
                pretty_str("κ", self.κ),
                pretty_str("Wm", self.Wm),
                pretty_str("Wc", self.Wc),
                pretty_str("subtract", self.subtract),
                pretty_str("sqrt", self.sqrt),
            ]
        )


class MerweScaledSigmas:
    """
    Generates sigma points and weights according to Van der Merwe's 2004 dissertation. It parametizes the sigma points using κ, α, β terms, and is the version seen in most publications.
    Unless you know better, this should be your default choice.
    """

    def __init__(
        self,
        n,
        kappa,
        alpha=1,
        beta=0,
        sqrt_method=None,
        subtract_method=None,
    ):
        self.n = n
        self.α = alpha
        self.β = beta
        self.κ = kappa if kappa is not None else 3 - n
        self.sqrt = linalg.cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract_method is None else subtract_method
        self.sigma_weights()

    def num_sigmas(self):
        """Number of sigma points for each variable in the state x"""
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        """Computes the sigma points for an UKF given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
        """

        if self.n != np.size(x):
            raise ValueError(f"expected size(x) {self.n}, but size is {np.size(x)}")

        if np.isscalar(x):
            x = np.asarray([x])

        n = self.n
        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)

        c = self.α**2 * (n + self.κ)  # scaling constant
        U = self.sqrt(c * P)  # upper-triangular matrix
        # λ = c - n

        sigmas = np.zeros((2 * n + 1, n))
        sigmas[0] = x
        for k in range(n):
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[n + k + 1] = self.subtract(x, U[k])

        # equivalent to
        # U = self.sqrt(P)
        # sigmas = np.hstack([np.zeros(M.shape), U, -U])
        # sigmas = np.sqrt(c) * sigmas + np.tile(M, sigmas.shape[1])

        return sigmas

    def sigma_weights(self):
        """Computes the weights for the scaled UKF."""

        n = self.n
        c = self.α**2 * (n + self.κ)  # scaling constant
        λ = c - n

        self.Wc = np.full(2 * n + 1, 0.5 / c)
        self.Wm = np.full(2 * n + 1, 0.5 / c)
        self.Wc[0] = λ / c + (1 - self.α**2 + self.β)
        self.Wm[0] = λ / c

    def sigma_mweights(self):
        """Returns the weights matrix for the scaled UKF."""

        Wm, Wc, c = self.sigma_weights()
        W = np.eye(len(Wc)) - np.tile(Wm, (1, len(Wm)))
        W = W @ np.diag(Wc) @ W.T

        return Wm, W, c

    def __repr__(self):
        return "\n".join(
            [
                "MerweScaledSigmas object",
                pretty_str("n", self.n),
                pretty_str("α", self.α),
                pretty_str("β", self.β),
                pretty_str("κ", self.κ),
                pretty_str("Wm", self.Wm),
                pretty_str("Wc", self.Wc),
                pretty_str("subtract", self.subtract),
                pretty_str("sqrt", self.sqrt),
            ]
        )


class SimplexSigmas:
    """
    Generates sigma points and weights according to the simplex
    method.
    """

    def __init__(self, n, α=1, sqrt_method=None, subtract=None):
        self.n = n
        self.α = α
        self.sqrt = linalg.cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        self.sigma_weights()

    def num_sigmas(self):
        """Number of sigma points for each variable in the state x"""
        return self.n + 1

    def sigma_points(self, x, P):
        """
        Computes the implex sigma points for an UKF given the mean (x) and covariance (P) of the filter.
        Returns tuple of the sigma points and weights.
        """

        if self.n != np.size(x):
            raise ValueError(f"expected size(x) {self.n}, but size is {np.size(x)}")

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])
        x = x.reshape(-1, 1)

        P = np.eye(n) * P if np.isscalar(P) else np.atleast_2d(P)
        U = self.sqrt(P)

        λ = n / (n + 1)
        Istar = np.array([[-1 / np.sqrt(2 * λ), 1 / np.sqrt(2 * λ)]])

        for d in range(2, n + 1):
            row = np.ones((1, Istar.shape[1] + 1)) * 1.0 / np.sqrt(λ * d * (d + 1))  # pylint: disable=unsubscriptable-object
            row[0, -1] = -d / np.sqrt(λ * d * (d + 1))
            Istar = np.r_[np.c_[Istar, np.zeros((Istar.shape[0]))], row]  # pylint: disable=unsubscriptable-object

        _I = np.sqrt(n) * Istar
        scaled_unitary = U.T @ _I

        sigmas = self.subtract(x, -scaled_unitary)
        return sigmas.T

    def sigma_weights(self):
        """Computes the weights for the scaled UKF."""

        n = self.n
        c = 1.0 / (n + 1)
        self.Wm = np.full(n + 1, c)
        self.Wc = self.Wm

    def __repr__(self):
        return "\n".join(
            [
                "SimplexSigmas object",
                pretty_str("n", self.n),
                pretty_str("α", self.α),
                pretty_str("Wm", self.Wm),
                pretty_str("Wc", self.Wc),
                pretty_str("subtract", self.subtract),
                pretty_str("sqrt", self.sqrt),
            ]
        )


class SphericalRadialSigmas:
    """Generates sigma points and weights according to the spherical radial method."""

    def __init__(self, n, P):
        self.n = n
        self.P = P

    def num_sigmas(self):
        """Number of sigma points for each variable in the state x"""

        return 2 * self.n

    def sigma_points(self, x, P):
        """Creates cubature points for the the specified state and covariance.

        Parameters
        ----------
        x : _type_
            _description_
        P : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        References
        ----------
        Arasaratnam, I, Haykin, S. "Cubature Kalman Filters," IEEE Transactions on Automatic Control, 2009, pp 1254-1269, vol 54, No 6
        """
        if self.n != np.size(x):
            raise ValueError(f"expected size(x) {self.n}, but size is {np.size(x)}")

        # dimension of P is
        n = self.n
        x = x.flatten()

        # evaluation points (nx2n)
        sigmas = np.empty((2 * n, n))
        U = linalg.cholesky(P) * np.sqrt(n)

        for k in range(n):
            sigmas[k] = x + U[k]
            sigmas[n + k] = x - U[k]

    def sigma_weights(self, x=None):
        n = self.num_sigmas(x)
        # weights are
        W = 1 / (2 * n)
        # weights
        W *= np.ones(2 * n)

from copy import deepcopy

import numpy as np
from numpy import linalg, random

from .helpers import pretty_str


class EnsembleKalmanFilter:
    """Implements the Ensemble Kalman Filter (EnKF).
    It works with both linear and nonlinear systems.

    References
    ----------
    John L Crassidis and John L. Junkins. "Optimal Estimation of Dynamic Systems. CRC Press, second edition. 2012. pp, 257-9.
    """

    def __init__(self, x, P, dim_z, dt, N, hx, fx):
        if dim_z <= 0:
            raise ValueError("dim_z must be greater than zero")
        if N <= 0:
            raise ValueError("N must be greater than zero")

        dim_x = len(x)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.N = N
        self.hx = hx
        self.fx = fx
        self.K = np.zeros((dim_x, dim_z))
        self.z = np.array([[None] * self.dim_z]).T
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        self.initialize(x, P)
        self.Q = np.eye(dim_x)  # process uncertainty
        self.R = np.eye(dim_z)  # state uncertainty

        # used to create error terms centered at 0 mean for
        # state and measurement
        self._mean = np.zeros(dim_x)
        self._mean_z = np.zeros(dim_z)

    def initialize(self, x, P):
        """Initializes the filter with the specified mean and covariance. Only need to call this when using the filter
        to filter more than one set of data.
        """

        if x.ndim != 1:
            raise ValueError("x must be a 1D np.array")

        self.sigmas = random.multivariate_normal(mean=x, cov=P, size=self.N)
        self.x = x
        self.P = P

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z, R=None):
        """Add a new measurement (z) to the Kalman filter. If z is None, nothing is changed."""

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if R is None:
            R = self.R
        if np.isscalar(R):
            R = np.eye(self.dim_z) * R

        N = self.N
        dim_z = len(z)
        sigmas_h = np.zeros((N, dim_z))

        # transform sigma points into measurement space
        for i in range(N):
            sigmas_h[i] = self.hx(self.sigmas[i])

        z_mean = np.mean(sigmas_h, axis=0)

        P_zz = (outer_product_sum(sigmas_h - z_mean) / (N - 1)) + R
        P_xz = outer_product_sum(self.sigmas - self.x, sigmas_h - z_mean) / (N - 1)

        self.S = P_zz
        self.SI = linalg.inv(self.S)
        self.K = P_xz @ self.SI

        e_r = random.multivariate_normal(self._mean_z, R, N)
        for i in range(N):
            self.sigmas[i] += self.K @ (z + e_r[i] - sigmas_h[i])

        self.x = np.mean(self.sigmas, axis=0)
        self.P = self.P - self.K @ self.S @ self.K.T

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self):
        """Predict next position."""

        N = self.N
        for i, s in enumerate(self.sigmas):
            self.sigmas[i] = self.fx(s, self.dt)

        e = random.multivariate_normal(self._mean, self.Q, N)
        self.sigmas += e

        self.x = np.mean(self.sigmas, axis=0)
        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def __repr__(self):
        return "\n".join(
            [
                "EnsembleKalmanFilter object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("dim_z", self.dim_z),
                pretty_str("dt", self.dt),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("K", self.K),
                pretty_str("S", self.S),
                pretty_str("sigmas", self.sigmas),
                pretty_str("hx", self.hx),
                pretty_str("fx", self.fx),
            ]
        )


def outer_product_sum(A, B=None):
    if B is None:
        B = A

    return np.sum(np.einsum("ij,ik->ijk", A, B), axis=0)

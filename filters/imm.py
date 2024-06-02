from copy import deepcopy
from itertools import product

import numpy as np

from .helpers import pretty_str


class IMMEstimator:
    """Implements an Interacting Multiple-Model (IMM) estimator.

    References
    ----------
    Bar-Shalom, Y., Li, X-R., and Kirubarajan, T. "Estimation with
    Application to Tracking and Navigation". Wiley-Interscience, 2001.

    Crassidis, J and Junkins, J. "Optimal Estimation of Dynamic Systems". CRC Press, second edition. 2012.
    """

    def __init__(self, filters, mu, M):
        self.num_f = len(filters)
        if self.num_f < 2:
            raise ValueError("filters must contain at least two filters")

        self.filters = filters
        self.mu = np.asarray(mu) / np.sum(mu)
        # transition matrix
        self.M = M

        x_shape = filters[0].x.shape
        for f in filters:
            if x_shape != f.x.shape:
                raise ValueError("All filters must have the same state dimension")

        self.x = np.zeros(filters[0].x.shape)
        self.P = np.zeros(filters[0].P.shape)
        self.N = len(filters)  # number of filters
        self.likelihood = np.zeros(self.N)
        self.omega = np.zeros((self.N, self.N))
        self._update_mixing_probabilities()

        # initialize IMM state estimate based on current filters
        self._update_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z):
        """Add a new measurement (z) to the Kalman filter. If z is None, nothing is changed."""

        # run update on each filter, and save the likelihood
        for i, f in enumerate(self.filters):
            f.update(z)
            self.likelihood[i] = f.likelihood

        # update mode probabilities from total probability * likelihood
        self.mu = self.cbar * self.likelihood
        self.mu /= np.sum(self.mu)  # normalize

        self._update_mixing_probabilities()

        # compute mixed IMM state and covariance and save posterior estimate
        self._update_state_estimate()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, u=None):
        """Predict next state (prior) using the IMM state propagation equations."""

        # compute mixed initial conditions
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):
            x = np.zeros(self.x.shape)
            for kf, wj in zip(self.filters, w):
                x += kf.x * wj
            xs.append(x)

            P = np.zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                P += wj * (np.outer(y, y) + kf.P)
            Ps.append(P)

        #  compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].copy()
            f.P = Ps[i].copy()
            f.predict(u)

        # compute mixed IMM state and covariance and save posterior estimate
        self._update_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def _update_state_estimate(self):
        """Computes the IMM's mixed state estimate from each filter using the the mode probability self.mu to weight the estimates."""
        self.x.fill(0)
        for f, mu in zip(self.filters, self.mu):
            self.x += f.x * mu

        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            y = f.x - self.x
            self.P += mu * (np.outer(y, y) + f.P)

    def _update_mixing_probabilities(self):
        """Compute the mixing probability for each filter."""

        self.cbar = self.mu @ self.M
        for i, j in product(range(self.N), range(self.N)):
            self.omega[i, j] = (self.M[i, j] * self.mu[i]) / self.cbar[j]

    def __repr__(self):
        return "\n".join(
            [
                "IMMEstimator object",
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("x_post", self.x_post),
                pretty_str("P_post", self.P_post),
                pretty_str("N", self.N),
                pretty_str("mu", self.mu),
                pretty_str("M", self.M),
                pretty_str("cbar", self.cbar),
                pretty_str("likelihood", self.likelihood),
                pretty_str("omega", self.omega),
            ]
        )


class MMAEFilterBank:
    """Implements the fixed Multiple Model Adaptive Estimator (MMAE).

    References
    ----------
    Zarchan and Musoff. "Fundamentals of Kalman filtering: A Practical
    Approach." AIAA, third edition.
    """

    def __init__(self, filters, p, dim_x, H=None):
        if len(filters) != len(p):
            raise ValueError("length of filters and p must be the same")

        if dim_x < 1:
            raise ValueError("dim_x must be >= 1")

        self.filters = filters
        self.p = np.asarray(p)
        self.dim_x = dim_x
        self.H = None if H is None else np.copy(H)
        # try to form a reasonable initial values, but good luck!
        try:
            self.z = np.copy(filters[0].z)
            self.x = np.copy(filters[0].x)
            self.P = np.copy(filters[0].P)

        except AttributeError:
            self.z = 0
            self.x = None
            self.P = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, u=0):
        """Predict next position using the Kalman filter state propagation equations for each filter in the bank."""

        for f in self.filters:
            f.predict(u)

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, H=None):
        """Add a new measurement (z) to the Kalman filter. If z is None, nothing is changed.

        Parameters
        ----------
        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this one call, otherwise  self.H will be used.
        """

        if H is None:
            H = self.H

        # new probability is recursively defined as prior * likelihood
        for i, f in enumerate(self.filters):
            f.update(z, R, H)
            self.p[i] *= f.likelihood

        self.p /= sum(self.p)  # normalize

        # compute estimated state and covariance of the bank of filters.
        self.P = np.zeros(self.filters[0].P.shape)

        # state can be in form [x,y,z,...] or [[x, y, z,...]].T
        is_row_vector = self.filters[0].x.ndim == 1
        self.x = np.zeros(self.dim_x) if is_row_vector else np.zeros((self.dim_x, 1))
        for f, p in zip(self.filters, self.p):
            self.x += f.x @ p
        for x, f, p in zip(self.x, self.filters, self.p):
            y = f.x - x
            self.P += p * (np.outer(y, y) + f.P)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def __repr__(self):
        return "\n".join(
            [
                "MMAEFilterBank object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("log-p", self.p),
            ]
        )

from copy import deepcopy

import numpy as np
from numpy import linalg

from .helpers import pretty_str


class HInfinityFilter:
    """Implements the H-Infinity filter."""

    def __init__(self, dim_x, dim_z, dim_u, gamma):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.gamma = gamma

        self.x = np.zeros((dim_x, 1))

        self.G = 0
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)

        self._VI = np.zeros((dim_z, dim_z))
        self._V = np.zeros((dim_z, dim_z))
        self.W = np.zeros((dim_x, dim_x))

        # gain and residual are computed during the innovation step.
        # We save them so that in case to inspect them for various purposes

        self.K = 0  # H-infinity gain
        self.y = np.zeros((dim_z, 1))
        self.z = np.zeros((dim_z, 1))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

    def update(self, z):
        """Add a new measurement `z` to the H-Infinity filter.

        Parameters
        ----------
        z : ndarray
            measurement for this update.
        """

        if z is None:
            return

        # rename for readability and a tiny extra bit of speed
        I_ = self._I
        gamma = self.gamma
        Q = self.Q
        H = self.H
        P = self.P
        x = self.x
        VI = self._VI
        F = self.F
        W = self.W

        # common subexpression H.T * V^-1
        HTVI = H.T @ VI

        L = linalg.inv(I_ - gamma * Q @ P + HTVI @ H @ P)

        # common subexpression P*L
        PL = P @ L

        K = F @ PL @ HTVI

        self.y = z - H @ x

        # x = x + Ky
        # predict new x with residual scaled by the H-Infinity gain
        self.x = self.x + K @ self.y
        self.P = F @ PL @ F.T + W

        # force P to be symmetric
        self.P = (self.P + self.P.T) / 2

        # pylint: disable=bare-except
        try:
            self.z = np.copy(z)
        except Exception:
            self.z = deepcopy(z)

    def predict(self, u=0):
        """Predict next position.

        Parameters
        ----------
        u : ndarray
            Optional control vector. If non-zero, it is multiplied by `B` to create the control input into the system.
        """

        # x = Fx + Gu
        self.x = self.F @ self.x + self.G @ u

    def batch_filter(self, Zs, update_first=False, saver=False):
        """Batch processes a sequences of measurements.

        Parameters
        ----------
        Zs : list-like
            list of measurements at each time step `self.dt` Missing
            measurements must be represented by 'None'.

        update_first : bool, default=False, optional,
            controls whether the order of operations is update followed by predict, or predict followed by update.

        saver : filterpy.common.Saver, optional.
            filterpy.common.Saver object. If provided, saver.save() will be called after every epoch

        Returns
        -------
        means: ndarray ((n, dim_x, 1))
            array of the state for each time step. Each entry is an np.array.
            In other words `means[k,:]` is the state at step `k`.

        covariance: ndarray((n, dim_x, dim_x))
            array of the covariances for each time step. In other words
            `covariance[k, :, :]` is the covariance at step `k`.
        """

        n = np.size(Zs, 0)

        # mean estimates from H-Infinity Filter
        means = np.zeros((n, self.dim_x, 1))

        # state covariances from H-Infinity Filter
        covariances = np.zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, z in enumerate(Zs):
                self.update(z)
                means[i, :] = self.x
                covariances[i, :, :] = self.P
                self.predict()

                if saver is not None:
                    saver.save()
        else:
            for i, z in enumerate(Zs):
                self.predict()
                self.update(z)

                means[i, :] = self.x
                covariances[i, :, :] = self.P

                if saver is not None:
                    saver.save()

        return (means, covariances)

    def residual_of(self, z):
        """Returns the residual for the given measurement (z). Does not alter the state of the filter."""
        return z - self.H @ self.x

    def measurement_of_state(self, x):
        """Helper function that converts a state into a measurement.

        Parameters
        ----------
        x : ndarray
            H-Infinity state vector

        Returns
        -------
        z : ndarray
            measurement corresponding to the given state
        """
        return self.H @ x

    @property
    def V(self):
        """Measurement noise matrix"""
        return self._V

    @V.setter
    def V(self, value):
        """Measurement noise matrix"""

        self._V = np.array([[value]], dtype=float) if np.isscalar(value) else value
        self._VI = linalg.inv(self._V)

    def __repr__(self):
        return "\n".join(
            [
                "HInfinityFilter object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("dim_z", self.dim_z),
                pretty_str("dim_u", self.dim_u),
                pretty_str("gamma", self.dim_u),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("V", self.V),
                pretty_str("W", self.W),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
            ]
        )

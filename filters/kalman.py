import math
import sys
from copy import deepcopy
from typing import Union

import numpy as np
from numpy import linalg
from scipy import stats

from .helpers import pretty_str
from .transformers import reshape_z


class KalmanFilter:
    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        dim_u: int = 0,
        _alpha_sq: float = 1.0,
    ):
        """Implementation of Kalman Filter.

        Parameters
        ----------
        dim_x : int
            Number of state variables for the Kalman filter.
            This is used to set the default size of P, Q, and u
        dim_z : int
            Number of of measurement inputs.
        dim_u : int (optional)
            Size of the control input, if it is being used.
            Default value 0 indicates it is not used.
        compute_log_likelihood : bool (default = True)
            Computes log likelihood by default, but this can be a slow
            computation, turn this computation off if you never use it.

        Attributes
        ----------
        x : numpy.array(dim_x, 1)
            Current state estimate.
        P : numpy.array(dim_x, dim_x)
            Current state covariance matrix.
        F : numpy.array()
            State Transition matrix.
        Q : numpy.array(dim_x, dim_x)
            Process noise covariance matrix.
        z : numpy.array
            Last measurement used in update(). Read only.
        H : numpy.array(dim_z, dim_x)
            Measurement function.
        R : numpy.array(dim_z, dim_z)
            Measurement noise covariance matrix.
        y : numpy.array
            Residual of the update step. Read only.
        K : numpy.array(dim_x, dim_z)
            Kalman gain of the update step. Read only.
        S :  numpy.array
            System uncertainty. Read only.
        log_likelihood : float
            log-likelihood of the last measurement. Read only.
        likelihood : float
            likelihood of last measurement. Read only.
        mahalanobis : float
            mahalanobis distance of the innovation. Read only.
        _alpha_sq : float
            Fading memory setting. 1.0 gives the normal Kalman filter, and values slightly larger than 1.0 (such as 1.02) give a fading memory effect. This formulation of the Fading memory filter is due to Dan Simon.

        References
        ----------
        Dan Simon. "Optimal State Estimation." John Wiley & Sons. p. 208-212. (2006)
        """
        if dim_x < 1:
            raise ValueError("dim_x ≥ 1")
        if dim_z < 1:
            raise ValueError("dim_z ≥ 1")
        if dim_u < 0:
            raise ValueError("dim_u ≥ 0")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # state
        self.x = np.zeros((dim_x, 1))
        # state covariance
        self.P = np.eye(dim_x)
        # transition matrix
        self.F = np.eye(dim_x)
        # transition covariance
        self.Q = np.eye(dim_x)

        # measurement
        self.z = np.array([[None] * self.dim_z]).T
        self.H = np.zeros((dim_z, dim_x))
        # measurement covariance
        self.R = np.eye(dim_z)

        # control matrix
        self.G = None
        # process-measurement cross correlation
        self.M = np.zeros((dim_x, dim_z))
        # fading memory control
        self._alpha_sq = _alpha_sq

        # gain and residual are computed during the innovation step
        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros((dim_z, 1))
        # system uncertainty
        self.S = np.zeros((dim_z, dim_z))
        # identity matrix.
        self._I = np.eye(dim_x)

        self.compute_log_likelihood = False
        # only computed only if requested via property
        if self.compute_log_likelihood:
            self._log_likelihood = math.log(sys.float_info.min)
            self._likelihood = sys.float_info.min
            self._mahalanobis = None

        # save the priors so that in case to inspect them for various purposes
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # save the posts so that in case to inspect them for various purposes
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(
        self,
        F: Union[bool, np.ndarray] = None,
        G: Union[bool, np.ndarray] = None,
        u: Union[bool, np.array] = None,
        Q: Union[bool, np.ndarray] = None,
    ):
        """Predict next state using the Kalman filter state propagation equations.

        Parameters
        ----------
        F : np.ndarray(dim_x, dim_x), or None, optional.
        G : np.ndarray(dim_x, dim_u), or None, optional.
        u : np.array, default 0, optional.
        Q : np.ndarray(dim_x, dim_x), scalar, or None, optional.
        """
        if F is None:
            F = self.F
        if G is None and u is not None:
            G = self.G
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # x = Fx + Gu
        if G is not None and u is not None:
            self.x = F @ self.x + G @ u
        else:
            self.x = F @ self.x
        # P = αFPF' + Q
        self.P = self._alpha_sq * (F @ self.P @ F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, H=None, R=None):
        """Add a new measurement (z) to the Kalman filter.

        Parameters
        ----------
        z : (dim_z, 1): array_like
        R : np.array, scalar, or None, optional.
        H : np.array, or None
        """
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
        else:
            z = reshape_z(z, self.dim_z, self.x.ndim)

        if H is None:
            H = self.H

        # y = z - Hx
        self.y = z - H @ self.x

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        # common subexpression for speed
        Pxz = self.P @ H.T
        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = H @ Pxz + R  # S = Pzz
        # K = PH'S^(-1)
        # map system uncertainty into Kalman gain
        self.K = Pxz @ linalg.inv(self.S)

        # x = x + Ky
        # predict new x with residual scaled by the Kalman gain
        self.x += self.K @ self.y

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable and works for non-optimal K
        I_KH = self._I - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def batch_filter(
        self,
        zs,
        Fs=None,
        Qs=None,
        Hs=None,
        Gs=None,
        us=None,
        Rs=None,
        update_first=False,
        saver=None,
    ):
        """Batch processes a sequences of measurements.

        Parameters
        ----------
        zs : list-like
            list of measurements at each time step `self.dt`.
        Fs : None, list-like, default=None, optional.
            If Fs is None then self.F is used for all epochs.
            Otherwise it must contain a list-like list of F's, one for
            each epoch. This allows you to have varying F per epoch.
        Qs : None, np.array or list-like, default=None, optional
             Behaves like Fs.
        Hs : None, np.array or list-like, default=None, optional
             Behaves like Fs.
        Rs : None, np.array or list-like, default=None, optional
             Behaves like Fs.
        Bs : None, np.array or list-like, default=None, optional
             Behaves like Fs.
        us : None, np.array or list-like, default=None, optional
             Behaves like Fs.
        update_first : bool, optional, default=False
        saver : helpers.Saver, optional
            If provided, saver.save() will be called after every epoch

        Returns
        -------
        means : np.array((n,dim_x,1))
            array of the state for each time step after the update. Each entry is an np.array. In other words `means[k,:]` is the state at step `k`.
        covariance : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.
        means_predictions : np.array((n,dim_x,1))
            array of the state for each time step after the predictions. Each entry is an np.array. In other words `means[k,:]` is the state at step `k`.
        covariance_predictions : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the prediction. In other words `covariance[k,:,:]` is the covariance at step `k`.
        """
        n = np.size(zs, 0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Gs is None:
            Gs = [self.G] * n
        if us is None:
            us = [0] * n
        if Rs is None:
            Rs = [self.R] * n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = np.zeros((n, self.dim_x))
            means_p = np.zeros((n, self.dim_x))
        else:
            means = np.zeros((n, self.dim_x, self.dim_z))
            means_p = np.zeros((n, self.dim_x, self.dim_z))

        # state covariances from Kalman Filter
        cov = np.zeros((n, self.dim_x, self.dim_x))
        cov_p = np.zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, G, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Gs, us)):
                self.update(z, H=H, R=R)
                means[i, :] = self.x
                cov[i, :, :] = self.P

                self.predict(u=u, G=G, F=F, Q=Q)
                means_p[i, :] = self.x
                cov_p[i, :, :] = self.P

                if saver is not None:
                    saver.save()
        else:
            for i, (z, F, Q, H, R, G, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Gs, us)):
                self.predict(u=u, G=G, F=F, Q=Q)
                means_p[i, :] = self.x
                cov_p[i, :, :] = self.P

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                cov[i, :, :] = self.P

                if saver is not None:
                    saver.save()

        return (means, cov, means_p, cov_p)

    def rts_smoother(self, Xs, Ps, Fs=None, Qs=None, inv=linalg.inv):
        """Runs the Rauch-Tung-Striebel Kalman smoother on a set of
        means and covariances computed by a Kalman filter. The usual input would come from the output of `batch_filter()`.
        """
        if len(Xs) != len(Ps):
            raise ValueError("length of Xs and Ps must be the same")

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n

        K = np.zeros((n, dim_x, dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n - 2, -1, -1):
            Pp[k] = Fs[k + 1] @ P[k] @ Fs[k + 1].T + Qs[k + 1]
            K[k] = P[k] @ Fs[k + 1].T @ inv(Pp[k])
            x[k] += K[k] @ (x[k + 1] - Fs[k + 1] @ x[k])
            P[k] += K[k] @ (P[k + 1] - Pp[k]) @ K[k].T

        return (x, P, K, Pp)

    def residual_of(self, z):
        """Returns the residual for the given measurement (z). Does not alter the state of the filter.

        Parameters
        ----------
        z : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        z = reshape_z(z, self.dim_z, self.x.ndim)
        return z - self.H @ self.x_prior

    def measurement_of_state(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        return self.H @ x

    @property
    def log_likelihood(self):
        """log-likelihood of the last measurement."""
        if self._log_likelihood is None:
            self._log_likelihood = stats.multivariate_normal.logpdf(
                x=self.y, cov=self.S
            )
        return self._log_likelihood

    @property
    def likelihood(self):
        """Computed from the log-likelihood. The log-likelihood can be very small, so by default we always return a number >= sys.float_info.min."""
        if self._likelihood is None:
            self._likelihood = np.exp(self.log_likelihood)
            if not np.isscalar(self._likelihood):
                self._likelihood = self._likelihood[0]
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def alpha(self):
        """Fading memory setting. 1.0 gives the normal Kalman filter, and values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates."""
        return self._alpha_sq**0.5

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) or value < 1:
            raise ValueError("alpha must be a float greater than 1")

        self._alpha_sq = value**2

    def log_likelihood_of(self, z):
        """
        log likelihood of the measurement `z`. This should only be called
        after a call to update(). Calling after predict() will yield an
        incorrect result."""

        if z is None:
            return math.log(sys.float_info.min)
        return stats.multivariate_normal.logpdf(z, self.H @ self.x, self.S)

    def __repr__(self):
        return "\n".join(
            [
                "KalmanFilter object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("dim_z", self.dim_z),
                pretty_str("dim_u", self.dim_u),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("x_post", self.x_post),
                pretty_str("P_post", self.P_post),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("H", self.H),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("S", self.S),
                pretty_str("M", self.M),
                pretty_str("B", self.B),
                pretty_str("z", self.z),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("likelihood", self.likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
                pretty_str("alpha", self.alpha),
            ]
        )


def imm_update(X_p, P_p, c_j, ind, dims, Y, H, R, nargout=5):
    """
    IMM_UPDATE  Interacting Multiple Model (IMM) Filter update step

    Syntax:
      [X_i,P_i,MU,X,P] = IMM_UPDATE(X_p,P_p,c_j,ind,dims,Y,H,R)

    In:
      X_p  - Cell array containing N^j x 1 mean state estimate vector for
             each model j after prediction step
      P_p  - Cell array containing N^j x N^j state covariance matrix for
             each model j after prediction step
      c_j  - Normalizing factors for mixing probabilities
      ind  - Indices of state components for each model as a cell array
      dims - Total number of different state components in the combined system
      Y    - Dx1 measurement vector.
      H    - Measurement matrices for each model as a cell array.
      R    - Measurement noise covariances for each model as a cell array.

    Out:
      X_i  - Updated state mean estimate for each model as a cell array
      P_i  - Updated state covariance estimate for each model as a cell array
      MU   - Estimated probabilities of each model
      X    - Combined state mean estimate
      P    - Combined state covariance estimate

    Description:
      IMM filter measurement update step.
    """

    # Number of models
    m = len(X_p)

    # Space for update state mean, covariance and likelihood of measurements
    X_i, P_i = np.empty((2, m), dtype=object)
    lbda = np.zeros(m)

    # Update for each model
    for i in range(m):
        # Update the state estimates
        X_i[i], P_i[i], _, _, _, lbda[i] = KalmanFilter.update(
            X_p[i], P_p[i], Y, H[i], R[i]
        )

    # Calculate the model probabilities
    MU = np.zeros(m)
    c = lbda @ c_j
    MU = c_j * lbda / c

    # Output the combined updated state mean and covariance, if wanted.
    if nargout > 3:
        # Space for estimates
        X = np.zeros((dims, 1))
        P = np.zeros((dims, dims))
        # Updated state mean
        for i in range(m):
            X[ind[i]] += MU[i] * X_i[i]

        # Updated state covariance
        for i in range(m):
            P[np.ix_(ind[i], ind[i])] += MU[i] * (
                P_i[i] + (X_i[i] - X[ind[i]]) @ (X_i[i] - X[ind[i]]).T
            )

    return X_i, P_i, MU, X, P


def imm_predict(X_ip, P_ip, MU_ip, p_ij, ind, dims, A, Q, nargout=3):
    """
    IMM_PREDICT  Interacting Multiple Model (IMM) Filter prediction step

    Syntax:
      [X_p,P_p,c_j,X,P] = IMM_PREDICT(X_ip,P_ip,MU_ip,p_ij,ind,dims,A,Q)

    In:
      X_ip  - Cell array containing N^j x 1 mean state estimate vector for
              each model j after update step of previous time step
      P_ip  - Cell array containing N^j x N^j state covariance matrix for
              each model j after update step of previous time step
      MU_ip - Vector containing the model probabilities at previous time step
      p_ij  - Model transition probability matrix
      ind   - Indexes of state components for each model as a cell array
      dims  - Total number of different state components in the combined system
      A     - State transition matrices for each model as a cell array.
      Q     - Process noise matrices for each model as a cell array.

    Out:
      X_p   - Predicted state mean for each model as a cell array
      P_p   - Predicted state covariance for each model as a cell array
      c_j   - Normalizing factors for mixing probabilities
      X     - Combined predicted state mean estimate
      P     - Combined predicted state covariance estimate

    Description:
      IMM filter prediction step.
    """

    # Number of models
    m = len(X_ip)

    # Normalizing factors for mixing probabilities
    c_j = MU_ip @ p_ij

    # Mixing probabilities
    MU_ij = p_ij * MU_ip[:, None] / c_j

    # Calculate the mixed state mean for each filter
    X_0j = np.empty(m, dtype=object)
    for j in range(m):
        X_0j[j] = np.zeros((dims, 1))
        for i in range(m):
            X_0j[j][ind[i]] += X_ip[i] * MU_ij[i, j]

    # Calculate the mixed state covariance for each filter
    P_0j = np.empty(m, dtype=object)
    for j in range(m):
        P_0j[j] = np.zeros((dims, dims))
        for i in range(m):
            P_0j[j][np.ix_(ind[i], ind[i])] += MU_ij[i, j] * (
                P_ip[i] + (X_ip[i] - X_0j[j][ind[i]]) @ (X_ip[i] - X_0j[j][ind[i]]).T
            )

    # Space for predictions
    X_p = np.empty(m, dtype=object)
    P_p = np.empty(m, dtype=object)

    # Make predictions for each model
    for i in range(m):
        X_p[i], P_p[i] = KalmanFilter.predict(
            X_0j[i][ind[i]], P_0j[i][np.ix_(ind[i], ind[i])], A[i], Q[i]
        )

    # Output the combined predicted state mean and covariance, if wanted
    if nargout > 3:
        # Space for estimates
        X = np.zeros((dims, 1))
        P = np.zeros((dims, dims))

        # Predicted state mean
        for i in range(m):
            X[ind[i]] += MU_ip[i] * X_p[i]
        # Predicted state covariance
        for i in range(m):
            P[np.ix_(ind[i], ind[i])] += MU_ip[i] * (
                P_p[i] + (X_ip[i] - X[ind[i]]) * (X_ip[i] - X[ind[i]]).T
            )

        return X_p, P_p, c_j, X, P

    return X_p, P_p, c_j


def imm_smooth(MM, PP, MM_i, PP_i, MU, p_ij, mu_0j, ind, dims, A, Q, R, H, Y):
    """
    IMM_SMOOTH   Fixed-interval IMM smoother using two IMM-filters.

    Syntax:
      [X_S,P_S,X_IS,P_IS,MU_S] = IMM_SMOOTH(MM,PP,MM_i,PP_i,MU,p_ij,mu_0j,ind,dims,A,Q,R,H,Y)

    In:
      MM    - NxK matrix containing the means of forward-time
              IMM-filter on each time step
      PP    - NxNxK matrix containing the covariances of forward-time
              IMM-filter on each time step
      MM_i  - Model-conditional means of forward-time IMM-filter on each time step
              as a cell array
      PP_i  - Model-conditional covariances of forward-time IMM-filter on each time
              step as a cell array
      MU    - Model probabilities of forward-time IMM-filter on each time step
      p_ij  - Model transition probability matrix
      mu_0j - Prior model probabilities
      ind   - Indices of state components for each model as a cell array
      dims  - Total number of different state components in the combined system
      A     - State transition matrices for each model as a cell array.
      Q     - Process noise matrices for each model as a cell array.
      R     - Measurement noise matrices for each model as a cell array.
      H     - Measurement matrices for each model as a cell array
      Y     - Measurement sequence


    Out:
      X_S  - Smoothed state means for each time step
      P_S  - Smoothed state covariances for each time step
      X_IS - Model-conditioned smoothed state means for each time step
      P_IS - Model-conditioned smoothed state covariances for each time step
      MU_S - Smoothed model probabilities for each time step

    Description:
      Two filter fixed-interval IMM smoother.
    """

    # Default values for mean and covariance
    MM_def = np.zeros((dims, 1))
    PP_def = np.diag(np.ones(dims))

    # Number of models
    m = len(A)

    # Number of measurements
    n = len(Y)

    # The prior model probabilities for each step
    p_jk = np.zeros((n, m))
    p_jk[0] = mu_0j
    for i1 in range(1, n):
        for i2 in range(m):
            p_jk[i1, i2] = np.sum(p_ij[i2] * p_jk[i1 - 1])

    # Backward-time transition probabilities
    p_ijb = np.zeros((n, m, m))
    for k in range(n):
        for i1 in range(m):
            # Normalizing constant
            b_i = np.sum(p_ij[i1] * p_jk[k])
            for j in range(m):
                p_ijb[k][i1, j] = p_ij[j, i1] * p_jk[k, j] / b_i

    # Space for overall smoothed estimates
    x_sk = np.zeros((n, dims, 1))
    P_sk = np.zeros((n, dims, dims))
    mu_sk = np.zeros((n, m))

    # Values of smoothed estimates at the last time step.
    x_sk[-1] = MM[-1]
    P_sk[-1] = PP[-1]
    mu_sk[-1] = MU[-1]

    # Space for model-conditioned smoothed estimates
    x_sik = np.empty((n, m), dtype=object)
    P_sik = np.empty((n, m), dtype=object)

    # Values for last time step
    x_sik[-1] = MM_i[-1]
    P_sik[-1] = PP_i[-1]

    # Backward-time estimated model probabilities
    mu_bp = MU[-1]

    # Space for model-conditioned backward-time updated means and covariances
    x_bki = MM_i[-1]
    P_bki = PP_i[-1]

    # Space for model-conditioned backward-time predicted means and covariances
    x_kp = np.tile(MM_def, (m, 1, 1))
    P_kp = np.tile(PP_def, (m, 1, 1))

    for k in range(n - 2, -1, -1):
        # Space for normalizing constants and conditional model probabilities
        a_j = np.zeros(m)
        mu_bijp = np.zeros((m, m))

        for i2 in range(m):
            # Normalizing constant
            a_j[i2] = np.sum(p_ijb[k][i2] * mu_bp)
            # Conditional model probability
            mu_bijp[:, i2] = p_ijb[k][i2] * mu_bp / a_j[i2]

            # Backward-time KF prediction step
            x_kp[i2][ind[i2]], P_kp[i2][np.ix_(ind[i2], ind[i2])] = (
                KalmanFilter.predict(x_bki[i2], P_bki[i2], linalg.inv(A[i2]), Q[i2])
            )

        # Space for mixed predicted mean and covariance
        x_kp0 = np.tile(MM_def, (m, 1, 1))
        P_kp0 = np.tile(PP_def, (m, 1, 1))

        # Space for measurement likelihoods
        lhood_j = np.zeros(m)

        for i2 in range(m):
            # Initialize with default values
            P_kp0[i2][np.ix_(ind[i2], ind[i2])] = np.zeros((len(ind[i2]), len(ind[i2])))

            # Mix the mean
            for i1 in range(m):
                x_kp0[i2][ind[i2]] += mu_bijp[i1, i2] * x_kp[i1][ind[i2]]

            # Mix the covariance
            for i1 in range(m):
                P_kp0[i2][np.ix_(ind[i2], ind[i2])] += mu_bijp[i1, i2] * (
                    P_kp[i1][np.ix_(ind[i2], ind[i2])]
                    + (x_kp[i1][ind[i2]] - x_kp0[i2][ind[i2]])
                    @ (x_kp[i1][ind[i2]] - x_kp0[i2][ind[i2]]).T
                )

            # Backward-time KF update
            (
                x_bki[i2][ind[i2]],
                P_bki[i2][np.ix_(ind[i2], ind[i2])],
                _,
                _,
                _,
                lhood_j[i2],
            ) = KalmanFilter.update(
                x_kp0[i2][ind[i2]],
                P_kp0[i2][np.ix_(ind[i2], ind[i2])],
                Y[k],
                H[i2],
                R[i2],
            )

        # Normalizing constant
        a = lhood_j @ a_j
        # Updated model probabilities
        mu_bp = a_j * lhood_j / a

        # Space for conditional measurement likelihoods
        lhood_ji = np.zeros((m, m))
        for i1 in range(m):
            for i2 in range(m):
                d_ijk = MM_def.copy()
                D_ijk = PP_def.copy()
                d_ijk += x_kp[i1]
                d_ijk[ind[i2]] -= MM_i[k, i2]
                PP2 = np.zeros((dims, dims))
                PP2[np.ix_(ind[i2], ind[i2])] = PP_i[k, i2]
                D_ijk = P_kp[i1] + PP2

                # Calculate the (approximate) conditional measurement likelihoods
                # D_ijk = 0.01^2*eye(size(D_ijk))
                lhood_ji[i2, i1], _ = stats.multivariate_normal.pdf(d_ijk, 0, D_ijk)

        d_j = np.zeros(m)
        for i2 in range(m):
            d_j[i2] = p_ij[i2] @ lhood_ji[i2]

        d = d_j @ MU[k]

        mu_ijsp = np.zeros((m, m))
        for i1 in range(m):
            for i2 in range(m):
                mu_ijsp[i1, i2] = p_ij[i2, i1] * lhood_ji[i2, i1] / d_j[i2]

        mu_sk[k] = d_j * MU[k] / d

        # Space for two-step conditional smoothing distributions p(x_k^j|m_{k+1}^i,y_{1:N}),
        # which are a products of two Gaussians
        x_jis = np.empty((m, m), dtype=object)
        P_jis = np.empty((m, m), dtype=object)
        for i2 in range(m):
            for i1 in range(m):
                MM1 = MM_def.copy()
                MM1[ind[i2]] = MM_i[k, i2]

                PP1 = PP_def.copy()
                PP1[np.ix_(ind[i2], ind[i2])] = PP_i[k, i2]

                # iPP1 = inv(PP1)
                # iPP2 = inv(P_kp[i1])

                # Covariance of the Gaussian product
                # P_jis{i2,i1} = inv(iPP1+iPP2)
                P_jis[i2, i1] = linalg.solve((PP1 + P_kp[i1]), PP1).T @ P_kp[i1]
                # Mean of the Gaussian product
                x_jis[i2, i1] = P_jis[i2, i1] @ (
                    linalg.solve(PP1, MM1) + linalg.solve(PP2, x_kp[i1])
                )

        # Mix the two-step conditional distributions to yield model-conditioned
        # smoothing distributions.
        for i2 in range(m):
            # Initialize with default values
            x_sik[k, i2] = MM_def.copy()
            P_sik[k, i2] = PP_def.copy()
            P_sik[k, i2][np.ix_(ind[i2], ind[i2])] = np.zeros(
                (len(ind[i2]), len(ind[i2]))
            )

            # Mixed mean
            for i1 in range(m):
                x_sik[k, i2] += mu_ijsp[i1, i2] * x_jis[i2, i1]

            # Mixed covariance
            for i1 in range(m):
                P_sik[k, i2] += mu_ijsp[i1, i2] * (
                    P_jis[i2, i1]
                    + (x_jis[i2, i1] - x_sik[k, i2]) @ (x_jis[i2, i1] - x_sik[k, i2]).T
                )

        # Mix the overall smoothed mean
        for i1 in range(m):
            x_sk[k] += mu_sk[k, i1] * x_sik[k, i1]

        # Mix the overall smoothed covariance
        for i1 in range(m):
            P_sk[k] += mu_sk[k, i1] * (
                P_sik[k, i1] + (x_sik[k, i1] - x_sk[k]) @ (x_sik[k, i1] - x_sk[k]).T
            )

    return x_sk, P_sk, x_sik, P_sik, mu_sk

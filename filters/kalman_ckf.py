import math
import sys
from copy import deepcopy

import numpy as np
from numpy import linalg
from scipy import stats

from .helpers import pretty_str
from .sigma_points import SphericalRadialSigmas


def ckf_packed_pc(x, fmmparam):
    """
    CKF_PACKED_PC - Pack P and C for the Cubature Kalman filter transform

    Syntax:
            pc = CKF_PACKED_PC(x,fmmparam)

    In:
            x - Evaluation point
            fmmparam - Array of handles and parameters to form the functions.

    Out:
            pc - Output values

    Description:
            Packs the integrals that need to be evaluated in nice function form to
            ease the evaluation. Evaluates P = (f-fm)(f-fm)' and C = (x-m)(f-fm)'.
    """

    f = fmmparam[0]
    m = fmmparam[1]
    fm = fmmparam[2]
    if len(fmmparam) >= 4:
        param = fmmparam[3]

    if type(f) == str or callable(f):
        F = f(x) if "param" not in locals() else f(x, param)
    elif type(f) == np.ndarray:
        F = f @ x
    else:
        F = f(x) if "param" not in locals() else f(x, param)
    d = x.shape[0]
    s = F.shape[0]

    # Compute P = (f-fm)(f-fm)' and C = (x-m)(f-fm)'
    # and form array of [vec(P):vec(C)]
    f_ = F.shape[1]
    pc = np.zeros((s**2 + d * s, f_))
    P = np.zeros((s, s))
    C = np.zeros((d, s))
    for k in range(f_):
        for j in range(s):
            for i in range(s):
                P[i, j] = (F[i, k] - fm[i]) * (F[j, k] - fm[j])
            for i in range(d):
                C[i, j] = (x[i, k] - m[i]) * (F[j, k] - fm[j])
        pc[:, k] = np.concatenate([P.reshape(s * s), C.reshape(s * d)])

    return pc


def ckf_transform2(m, P, g, param=None, varargin=None):
    """
    CKF_TRANSFORM - Cubature Kalman filter transform of random variables

    Syntax:
      [mu,S,C,SX,W] = CKF_TRANSFORM(M,P,g,param)

    In:
      M - Random variable mean (Nx1 column vector)
      P - Random variable covariance (NxN pos.def. matrix)
      g - Transformation function of the form g(x,param) as
          matrix, inline function, function name or function reference
      g_param - Parameters of g               (optional, default empty)

    Out:
      mu - Estimated mean of y
       S - Estimated covariance of y
       C - Estimated cross-covariance of x and y
      SX - Sigma points of x
       W - Weights as cell array
    """

    # Estimate the mean of g
    if param is None:
        mu, SX, W, _ = SphericalRadialSigmas(g, m, P)
    else:
        mu, SX, W, _ = SphericalRadialSigmas(g, m, P, param)

    # Estimate the P and C
    if param is None:
        pc, SX, W, _ = SphericalRadialSigmas(ckf_packed_pc, m, P, [g, m, mu])
    else:
        pc, SX, W, _ = SphericalRadialSigmas(ckf_packed_pc, m, P, [g, m, mu, param])

    d = m.shape[0]
    s = mu.shape[0]
    S = pc[: s**2].reshape((s, s))
    C = pc[s**2 :].reshape((d, s))

    return mu, S, C, SX, W


def ckf_transform(Xs, Q):
    """_summary_

    Parameters
    ----------
    Xs : _type_
        _description_
    Q : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    m, n = Xs.shape

    x = sum(Xs, 0)[:, None] / m
    P = np.zeros((n, n))
    xf = x.flatten()
    for k in range(m):
        P += np.outer(Xs[k], Xs[k]) - np.outer(xf, xf)

    P *= 1 / m
    P += Q

    return x, P


class CubatureKalmanFilter:
    """Implements the Cubuture Kalman filter (CKF).

    References
    ----------
    Arasaratnam, I, Haykin, S. "Cubature Kalman Filters," IEEE Transactions on Automatic Control, 2009, pp 1254-1269, vol 54, No 6
    """

    def _init_(
        self,
        dim_x,
        dim_z,
        dt,
        hx,
        fx,
        x_mean_fn=None,
        z_mean_fn=None,
        residual_x=None,
        residual_z=None,
    ):
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.K = 0
        self.dim_x = dim_x
        self.dim_z = dim_z
        self._dt = dt
        self._num_sigmas = 2 * dim_x
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn
        self.y = 0
        self.z = np.array([[None] * self.dim_z]).T
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        self.residual_x = np.subtract if residual_x is None else residual_x
        self.residual_z = np.subtract if residual_z is None else residual_z
        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update
        self.sigmas_f = np.zeros((2 * self.dim_x, self.dim_x))
        self.sigmas_h = np.zeros((2 * self.dim_x, self.dim_z))

        # Only computed only if requested via property
        self._log_likelihood = math.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, dt=None, fx_args=()):
        """Performs the predict step of the CKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P).

        Important: this MUST be called before update() is called for the first time.
        """

        if dt is None:
            dt = self._dt

        if not isinstance(fx_args, tuple):
            fx_args = (fx_args,)

        sigmas = SphericalRadialSigmas(self.x, self.P)

        # evaluate cubature points
        for k in range(self._num_sigmas):
            self.sigmas_f[k] = self.fx(sigmas[k], dt, *fx_args)

        self.x, self.P = ckf_transform(self.sigmas_f, self.Q)

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, hx_args=()):
        """Update the CKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.
        """

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        for k in range(self._num_sigmas):
            self.sigmas_h[k] = self.hx(self.sigmas_f[k], *hx_args)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = ckf_transform(self.sigmas_h, R)
        self.SI = linalg.inv(self.S)

        # compute cross variance of the state and the measurements
        m = self._num_sigmas  # literaure uses m for scaling factor
        xf = self.x.flatten()
        zpf = zp.flatten()

        Pxz = (
            np.sum(
                np.einsum(
                    "ij,ik->ijk",
                    self.sigmas_f - xf,
                    self.sigmas_h - zpf,
                ),
                axis=0,
            )
            / m
        )

        self.K = Pxz @ self.SI  # Kalman gain
        self.y = self.residual_z(z, zp)  # residual

        self.x += self.K @ self.y
        self.P -= self.K @ self.S @ self.K.T

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    @property
    def log_likelihood(self):
        """
        Computed the log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = stats.multivariate_normal.logpdf(
                x=self.y, cov=self.S
            )
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small, so by default we always return a number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = math.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """ "
        Mahalanobis distance of innovation.
        e.g. 3 means measurement was 3 standard deviations away from the predicted value.
        """
        if self._mahalanobis is None:
            self._mahalanobis = math.sqrt(float(self.y.T @ self.SI @ self.y))
        return self._mahalanobis

    def _repr_(self):
        return "\n".join(
            [
                "CubatureKalmanFilter object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("dim_z", self.dim_z),
                pretty_str("dt", self._dt),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("likelihood", self.likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
            ]
        )


def crts_smooth(M, P, f, Q, f_param=None, same_p=True):
    """
    CRTS_SMOOTH - Additive form cubature Rauch-Tung-Striebel smoother

    Syntax:
            [M,P,D] = CKF_SMOOTH(M,P,a,Q,[param,same_p])

    In:
            M - NxK matrix of K mean estimates from Cubature Kalman filter
            P - NxNxK matrix of K state covariances from Cubature Kalman Filter
            f - Dynamic model function as a matrix F defining
                    linear function f(x) = F*x, inline function,
                    function handle or name of function in
                    form f(x,param)                   (optional, default eye())
            Q - NxN process noise covariance matrix or NxNxK matrix
                    of K state process noise covariance matrices for each step.
            f_param - Parameters of f. Parameters should be a single cell array,
                            vector or a matrix containing the same parameters for each
                            step, or if different parameters are used on each step they
                            must be a cell array of the format { param_1, param_2, ...},
                            where param_x contains the parameters for step x as a cell array,
                            a vector or a matrix.   (optional, default empty)
            same_p - If 1 uses the same parameters
                            on every time step      (optional, default 1)

    Out:
            M - Smoothed state mean sequence
            P - Smoothed state covariance sequence
            D - Smoother gain sequence

    Description:
            Cubature Rauch-Tung-Striebel smoother algorithm. Calculate
            "smoothed" sequence from given Kalman filter output sequence by
            conditioning all steps to all measurements. Uses the spherical-
            radial cubature rule.
    """
    M = M.copy()
    P = P.copy()

    # Apply defaults
    m_1, m_2 = M.shape[:2]

    if f is None:
        f = np.eye(m_2)

    if Q is None:
        Q = np.zeros(m_2)

    # Extend Q if NxN matrix
    if len(Q.shape) == 2:
        Q = np.tile(Q, (m_1, 1, 1))

    # Run the smoother
    D = np.zeros((m_1, m_2, m_2))
    if f_param is None:
        for k in range(m_1 - 2, -1, -1):
            m_pred, P_pred, C, *_ = ckf_transform(M[k], P[k], f)
            P_pred += Q[k]
            D[k] = linalg.solve(P_pred.T, C.T).T
            M[k] += D[k] @ (M[k + 1] - m_pred)
            P[k] += D[k] @ (P[k + 1] - P_pred) @ D[k].T
    else:
        for k in range(m_1 - 2, -1, -1):
            if f_param is None:
                params = None
            elif same_p:
                params = f_param
            else:
                params = f_param[k]

            m_pred, P_pred, C, *_ = ckf_transform(M[k], P[k], f, params)
            P_pred += Q[k]
            D[k] = linalg.solve(P_pred.T, C.T).T
            M[k] += D[k] @ (M[k + 1] - m_pred)
            P[k] += D[k] @ (P[k + 1] - P_pred) @ D[k].T

    return M, P, D

import math
import sys
from copy import deepcopy

import numpy as np
from numpy import linalg
from scipy import stats

from .helpers import pretty_str
from .kalman import KalmanFilter


def unscented_transform(
    sigmas,
    Wm,
    Wc,
    noise_cov=None,
    mean_fn=None,
    residual_fn=None,
):
    """Computes unscented transform of a set of sigma points and weights.

    Parameters
    ----------
    sigmas : ndarray, of size (n, 2n+1)
        2D array of sigma points.
    Wm : ndarray
        Weights for the mean.
    Wc : ndarray
        Weights for the covariance.
    noise_cov : ndarray, optional
        noise matrix added to the final computed covariance matrix.
    mean_fn : callable (sigma_points, weights), optional
        Function that computes the mean of the provided sigma points
        and weights. Use this when your state variable contains nonlinear values such as angles which cannot be summed.
    residual_fn : callable (x, y), optional
        Function that computes the residual (difference) between x and y.
        Supply this when your state variable cannot support subtraction, such as angles. x and y are state vectors, not scalars.

    Returns
    -------
    x : ndarray
        Mean of the sigma points after passing through the transform.
    P : ndarray
        covariance of the sigma points after passing through the transform.
    """

    kmax, n = sigmas.shape

    try:
        x = Wm @ sigmas if mean_fn is None else mean_fn(sigmas, Wm)
    except:
        print(sigmas)
        raise

    # new covariance is the sum of the outer product of the residuals
    # times the weights
    # the fast way: for linear cases
    if residual_fn is np.subtract or residual_fn is None:
        y = sigmas - x[np.newaxis, :]
        P = y.T @ np.diag(Wc) @ y
    else:
        # the slow way: for nonlinear cases, like angles
        P = np.zeros((n, n))
        for k in range(kmax):
            y = residual_fn(sigmas[k], x)
            P += Wc[k] * np.outer(y, y)

    if noise_cov is not None:
        P += noise_cov

    return (x, P)


class UnscentedKalmanFilter:
    def __init__(
        self,
        dim_x,
        dim_z,
        dt,
        hx,
        fx,
        points,
        sqrt_fn=None,
        x_mean_fn=None,
        z_mean_fn=None,
        residual_x=None,
        residual_z=None,
        state_add=None,
    ):
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self._dim_x = dim_x
        self._dim_z = dim_z
        self.points_fn = points
        self._dt = dt
        self._num_sigmas = points.num_sigmas()
        self.hx = hx
        self.fx = fx
        self.x_mean = x_mean_fn
        self.z_mean = z_mean_fn

        # only computed only if requested via property
        self._log_likelihood = math.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.msqrt = linalg.cholesky if sqrt_fn is None else sqrt_fn
        # weights for the means and covariances.
        self.Wm, self.Wc = points.Wm, points.Wc

        self.residual_x = np.subtract if residual_x is None else residual_x
        self.residual_z = np.subtract if residual_z is None else residual_z
        self.state_add = np.add if state_add is None else state_add
        # sigma points transformed through f(x) and h(x)
        # variables for efficiency so we don't recreate every update

        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, self._dim_z))

        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros((dim_z))
        self.z = np.array([[None] * dim_z]).T
        self.S = np.zeros((dim_z, dim_z))
        self.SI = np.zeros((dim_z, dim_z))

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)

        # pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(
            self.sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x
        )

        # update sigma points to reflect the new variance of the points
        self.sigmas_f = self.points_fn.sigma_points(self.x, self.P)

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        if z is None:
            self.z = np.array([[None] * self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self._dim_z) * R

        sigmas_h = [hx(s, **hx_args) for s in self.sigmas_f]
        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(
            self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z
        )
        self.SI = linalg.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)

        self.K = Pxz @ self.SI  # Kalman gain
        self.y = self.residual_z(z, zp)  # residual

        # update Gaussian state estimate (x, P)
        self.x = self.state_add(self.x, self.K @ self.y)
        self.P -= self.K @ self.S @ self.K.T

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        """
        Compute cross variance of the state `x` and measurement `z`.
        """

        Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
        N = sigmas_f.shape[0]
        for i in range(N):
            dx = self.residual_x(sigmas_f[i], x)
            dz = self.residual_z(sigmas_h[i], z)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        return Pxz

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        """Computes the values of sigmas_f. It is useful when you need to call update more than once between calls to predict, so the sigmas correctly reflect the updated state x, P."""

        if fx is None:
            fx = self.fx

        # calculate sigma points for given mean and covariance
        sigmas = self.points_fn.sigma_points(self.x, self.P)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = fx(s, dt, **fx_args)

    def batch_filter(self, zs, Rs=None, dts=None, UT=None, saver=None):
        try:
            z = zs[0]
        except TypeError as e:
            raise TypeError("zs must be list-like") from e

        if self._dim_z == 1:
            if not np.isscalar(z) and (z.ndim != 1 or len(z) != 1):
                raise TypeError("zs must be a list of scalars or 1D, 1 element arrays")
        elif len(z) != self._dim_z:
            raise TypeError(
                f"each element in zs must be a 1D array of length {self._dim_z}"
            )

        z_n = len(zs)

        if Rs is None:
            Rs = [self.R] * z_n

        if dts is None:
            dts = [self._dt] * z_n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = np.zeros((z_n, self._dim_x))
        else:
            means = np.zeros((z_n, self._dim_x, 1))

        # state covariances from Kalman Filter
        covariances = np.zeros((z_n, self._dim_x, self._dim_x))

        for i, (z, r, dt) in enumerate(zip(zs, Rs, dts)):
            self.predict(dt=dt, UT=UT)
            self.update(z, r, UT=UT)
            means[i, :] = self.x
            covariances[i, :, :] = self.P

            if saver is not None:
                saver.save()

        return (means, covariances)

    def rts_smoother(self, Xs, Ps, Qs=None, dts=None, UT=None):
        """Runs the Rauch-Tung-Striebel Kalman smoother on a set of
        means and covariances computed by the UKF. The usual input
        would come from the output of `batch_filter()`.
        """

        if len(Xs) != len(Ps):
            raise ValueError("Xs and Ps must have the same length")

        n, dim_x = Xs.shape

        if dts is None:
            dts = [self._dt] * n
        elif np.isscalar(dts):
            dts = [dts] * n

        if Qs is None:
            Qs = [self.Q] * n

        if UT is None:
            UT = unscented_transform

        # smoother gain
        Ks = np.zeros((n, dim_x, dim_x))

        num_sigmas = self._num_sigmas

        xs, ps = Xs.copy(), Ps.copy()
        sigmas_f = np.zeros((num_sigmas, dim_x))

        for k in reversed(range(n - 1)):
            # create sigma points from state estimate, pass through state func
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(num_sigmas):
                sigmas_f[i] = self.fx(sigmas[i], dts[k])

            xb, Pb = UT(
                sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean, self.residual_x
            )

            # compute cross variance
            Pxb = 0
            for i in range(num_sigmas):
                y = self.residual_x(sigmas_f[i], xb)
                z = self.residual_x(sigmas[i], Xs[k])
                Pxb += self.Wc[i] * np.outer(z, y)

            # compute gain
            K = Pxb @ linalg(Pb)

            # update the smoothed estimates
            xs[k] += K @ self.residual_x(xs[k + 1], xb)
            ps[k] += K @ (ps[k + 1] - Pb) @ K.T
            Ks[k] = K

        return (xs, ps, Ks)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = stats.multivariate_normal.logpdf(
                self.y, cov=self.S, allow_singular=True
            )
        return self._log_likelihood

    @property
    def likelihood(self):
        """Computed from the log-likelihood. The log-likelihood can be very small, so by default we always return a number >= sys.float_info.min."""
        if self._likelihood is None:
            self._likelihood = math.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(self.y.T @ self.SI @ self.y))
        return self._mahalanobis

    def __repr__(self):
        return "\n".join(
            [
                "UnscentedKalmanFilter object",
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("S", self.S),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("likelihood", self.likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
                pretty_str("sigmas_f", self.sigmas_f),
                pretty_str("h", self.sigmas_h),
                pretty_str("Wm", self.Wm),
                pretty_str("Wc", self.Wc),
                pretty_str("residual_x", self.residual_x),
                pretty_str("residual_z", self.residual_z),
                pretty_str("msqrt", self.msqrt),
                pretty_str("hx", self.hx),
                pretty_str("fx", self.fx),
                pretty_str("x_mean", self.x_mean),
                pretty_str("z_mean", self.z_mean),
            ]
        )


def utf_smooth2(
    M,
    P,
    Y,
    ia=None,
    Q=None,
    aparam=None,
    h=None,
    R=None,
    hparam=None,
    alpha=None,
    beta=None,
    kappa=None,
    mat=0,
    same_p_a=1,
    same_p_h=1,
):
    """
    UTF_SMOOTH1  Smoother based on two unscented Kalman filters

    Syntax:
      [M,P] = UTF_SMOOTH1(M,P,Y,[ia,Q,aparam,h,R,hparam,,alpha,beta,kappa,mat,same_p_a,same_p_h])

    In:
      M - NxK matrix of K mean estimates from Kalman filter
      P - NxNxK matrix of K state covariances from Kalman Filter
      Y - Measurement vector
     ia - Inverse prediction as a matrix IA defining
          linear function ia(xw) = IA*xw, inline function,
          function handle or name of function in
          form ia(xw,param)                         (optional, default eye())
      Q - Process noise of discrete model           (optional, default zero)
      aparam - Parameters of a                      (optional, default empty)
      h  - Measurement model function as a matrix H defining
           linear function h(x) = H*x, inline function,
           function handle or name of function in
           form h(x,param)
      R  - Measurement noise covariance.
      hparam - Parameters of h              (optional, default aparam)
      alpha - Transformation parameter      (optional)
      beta  - Transformation parameter      (optional)
      kappa - Transformation parameter      (optional)
      mat   - If 1 uses matrix form         (optional, default 0)
      same_p_a - If 1 uses the same parameters
                 on every time step for a   (optional, default 1)
      same_p_h - If 1 uses the same parameters
                 on every time step for h   (optional, default 1)

    Out:
      M - Smoothed state mean sequence
      P - Smoothed state covariance sequence

    Description:
      Two filter nonlinear smoother algorithm. Calculate "smoothed"
      sequence from given extended Kalman filter output sequence
      by conditioning all steps to all measurements.
    """
    M = M.copy()
    P = P.copy()

    m_1 = M.shape[0]

    # Run the backward filter
    BM = np.zeros(M.shape)
    BP = np.zeros(P.shape)
    # fm = zeros(size(M,1),1)
    # fP = 1e12*eye(size(M,1))

    fm = M[-1]
    fP = P[-1]
    BM[-1] = fm
    BP[-1] = fP
    for k in range(m_1 - 2, -1, -1):
        if hparam is None:
            hparams = None
        elif same_p_h:
            hparams = hparam
        else:
            hparams = hparam[k]

        if aparam is None:
            aparams = None
        elif same_p_a:
            aparams = aparam
        else:
            aparams = aparam[k]

        fm, fP, *_ = UnscentedKalmanFilter().update(
            fm, fP, Y[k + 1], h, R, hparams, alpha, beta, kappa, mat
        )

        # Backward prediction
        fm, fP = UnscentedKalmanFilter().predict(fm, fP, ia, Q, aparams)
        BM[k] = fm
        BP[k] = fP

    # Combine estimates
    for k in range(m_1 - 1):
        tmp = linalg.inv(linalg.inv(P[k]) + linalg.inv(BP[k]))
        M[k] = tmp @ (linalg.solve(P[k], M[k]) + linalg.solve(BP[k], BM[k]))
        P[k] = tmp

    return M, P


def urts_smooth1(
    M, P, f, Q, f_param=None, alpha=None, beta=None, kappa=None, mat=0, same_p=True
):
    """
    URTS_SMOOTH1  Additive form Unscented Rauch-Tung-Striebel smoother

    Syntax:
      [M,P,D] = URTS_SMOOTH1(M,P,f,Q,[f_param,alpha,beta,kappa,mat,same_p])

    In:
      M - NxK matrix of K mean estimates from Unscented Kalman filter
      P - NxNxK matrix of K state covariances from Unscented Kalman Filter
      f - Dynamic model function as a matrix A defining
          linear function f(x) = A*x, inline function,
          function handle or name of function in
          form a(x,param)                   (optional, default eye())
      Q - NxN process noise covariance matrix or NxNxK matrix
          of K state process noise covariance matrices for each step.
      f_param - Parameters of f. Parameters should be a single cell array,
              vector or a matrix containing the same parameters for each
              step, or if different parameters are used on each step they
              must be a cell array of the format { param_1, param_2, ...},
              where param_x contains the parameters for step x as a cell array,
              a vector or a matrix.   (optional, default empty)
      alpha - Transformation parameter      (optional)
      beta  - Transformation parameter      (optional)
      kappa - Transformation parameter      (optional)
      mat   - If 1 uses matrix form         (optional, default 0)
      same_p - If 1 uses the same parameters
               on every time step      (optional, default 1)


    Out:
      M - Smoothed state mean sequence
      P - Smoothed state covariance sequence
      D - Smoother gain sequence

    Description:
      Unscented Rauch-Tung-Striebel smoother algorithm. Calculate
      "smoothed" sequence from given Kalman filter output sequence by
      conditioning all steps to all measurements.
    """
    M = M.copy()
    P = P.copy()

    m_1 = M.shape[0]
    m_2 = M.shape[1]

    # Apply defaults
    if f is None:
        f = np.eye(m_2)

    if Q is None:
        Q = np.zeros(m_2)

    # Extend Q if NxN matrix
    if len(Q.shape) < 3:
        Q = np.tile(Q, (m_1, 1, 1))

    # Run the smoother
    D = np.zeros((m_1, m_2, m_2))
    for k in range(m_1 - 2, -1, -1):
        if f_param is None:
            params = None
        elif same_p:
            params = f_param
        else:
            params = f_param[k]

        tr_param = [alpha, beta, kappa, mat]
        m_pred, P_pred, C, *_ = unscented_transform(M[k], P[k], f, params, tr_param)
        P_pred += Q[k]
        D[k] = linalg.solve(P_pred.T, C.T).T
        M[k] += D[k] @ (M[k + 1] - m_pred)
        P[k] += D[k] @ (P[k + 1] - P_pred) @ D[k].T

    return M, P, D


def uimm_predict(X_ip, P_ip, MU_ip, p_ij, ind, dims, A, a, param, Q, nargout=3):
    """
    IMM_PREDICT  UKF based Interacting Multiple Model (IMM) Filter prediction step

    Syntax:
      [X_p,P_p,c_j,X,P] = UIMM_PREDICT(X_ip,P_ip,MU_ip,p_ij,ind,dims,A,a,param,Q)

    In:
      X_ip  - Cell array containing N^j x 1 mean state estimate vector for
              each model j after update step of previous time step
      P_ip  - Cell array containing N^j x N^j state covariance matrix for
              each model j after update step of previous time step
      MU_ip - Vector containing the model probabilities at previous time step
      p_ij  - Model transition matrix
      ind   - Indices of state components for each model as a cell array
      dims  - Total number of different state components in the combined system
      A     - Dynamic model matrices for each linear model as a cell array
      a     - Dynamic model functions for each non-linear model
      param - Parameters of a
      Q     - Process noise matrices for each model as a cell array.

    Out:
      X_p  - Predicted state mean for each model as a cell array
      P_p  - Predicted state covariance for each model as a cell array
      c_j  - Normalizing factors for mixing probabilities
      X    - Combined predicted state mean estimate
      P    - Combined predicted state covariance estimate

    Description:
      IMM-UKF filter prediction step. If some of the models have linear
      dynamics standard Kalman filter prediction step is used for those.

    See also:
      UIMM_UPDATE, UIMM_SMOOTH

    History:
      01.11.2007 JH The first official version.



    $Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $

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
        if a is None or a[i] is None:
            X_p[i], P_p[i] = KalmanFilter().predict(
                X_0j[i][ind[i]], P_0j[i][np.ix_(ind[i], ind[i])], A[i], Q[i]
            )
        else:
            X_p[i], P_p[i], _ = UnscentedKalmanFilter().predict(
                X_0j[i][ind[i]], P_0j[i][np.ix_(ind[i], ind[i])], a[i], Q[i], param[i]
            )

    # Output the combined predicted state mean and covariance, if wanted.
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


def uimm_update(X_p, P_p, c_j, ind, dims, Y, H, h, R, param, nargout=5):
    """
    IMM_UPDATE  UKF based Interacting Multiple Model (IMM) Filter update step

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
      h    - Measurement mean
      param - parameters
      R    - Measurement noise covariances for each model as a cell array.

    Out:
      X_i  - Updated state mean estimate for each model as a cell array
      P_i  - Updated state covariance estimate for each model as a cell array
      MU   - Probabilities of each model
      X    - Combined state mean estimate
      P    - Combined state covariance estimate

    Description:
      IMM-UKF filter measurement update step. If some of the models have linear
      measurements standard Kalman filter update step is used for those.

    See also:
      IMM_PREDICT, IMM_SMOOTH, IMM_FILTER

    History:
      01.11.2007 JH The first official version.

    Copyright (C) 2007 Jouni Hartikainen

    $Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $

    This software is distributed under the GNU General Public
    Licence (version 2 or later); please refer to the file
    Licence.txt, included with the software, for details.
    """

    # Number of models
    m = len(X_p)

    # Space for update state mean, covariance and likelihood of measurements
    X_i = np.empty(m, dtype=object)
    P_i = np.empty(m, dtype=object)
    lbda = np.zeros(m)

    # Update for each model
    for i in range(m):
        # Update the state estimates
        if h is None or h[i] is None:
            X_i[i], P_i[i], _, _, _, lbda[i] = KalmanFilter().update(
                X_p[i], P_p[i], Y, H[i], R[i]
            )
        else:
            X_i[i], P_i[i], _, _, _, lbda[i] = UnscentedKalmanFilter().update(
                X_p[i], P_p[i], Y, h[i], R[i], param[i]
            )

    # Calculate the model probabilities
    MU = np.zeros(m)
    c = np.sum(lbda * c_j)
    MU = c_j * lbda / c

    # In case lbda's happen to be zero
    if c == 0:
        MU = c_j

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
                P_i[i] + (X_i[i] - X[ind[i]]) * (X_i[i] - X[ind[i]]).T
            )

        return X_i, P_i, MU, X, P

    return X_i, P_i, MU


def uimm_smooth(
    MM,
    PP,
    MM_i,
    PP_i,
    MU,
    p_ij,
    mu_0j,
    ind,
    dims,
    A,
    a,
    a_param,
    Q,
    R,
    H,
    h,
    h_param,
    Y,
):
    """
    UIMM_SMOOTH   UKF based Fixed-interval IMM smoother using two IMM-UKF filters.

    Syntax:
      [X_S,P_S,X_IS,P_IS,MU_S] = UIMM_SMOOTH(MM,PP,MM_i,PP_i,MU,p_ij,mu_0j,ind,dims,A,a,a_param,Q,R,H,h,h_param,Y)

    In:
      MM    - Means of forward-time IMM-filter on each time step
      PP    - Covariances of forward-time IMM-filter on each time step
      MM_i  - Model-conditional means of forward-time IMM-filter on each time step
      PP_i  - Model-conditional covariances of forward-time IMM-filter on each time step
      MU    - Model probabilities of forward-time IMM-filter on each time step
      p_ij  - Model transition probability matrix
      ind   - Indices of state components for each model as a cell array
      dims  - Total number of different state components in the combined system
      A     - Dynamic model matrices for each linear model and Jacobians of each
              non-linear model's measurement model function as a cell array
      a     - Cell array containing function handles for dynamic functions
              for each model having non-linear dynamics
      a_param - Parameters of a as a cell array.
      Q     - Process noise matrices for each model as a cell array.
      R     - Measurement noise matrices for each model as a cell array.
      H     - Measurement matrices for each linear model and Jacobians of each
              non-linear model's measurement model function as a cell array
      h     - Cell array containing function handles for measurement functions
              for each model having non-linear measurements
      h_param - Parameters of h as a cell array.
      Y     - Measurement sequence

    Out:
      X_S  - Smoothed state means for each time step
      P_S  - Smoothed state covariances for each time step
      X_IS - Model-conditioned smoothed state means for each time step
      P_IS - Model-conditioned smoothed state covariances for each time step
      MU_S - Smoothed model probabilities for each time step

    Description:
      UKF based two-filter fixed-interval IMM smoother.

    See also:
      UIMM_UPDATE, UIMM_PREDICT

    History:
      09.01.2008 JH The first official version.

    Copyright (C) 2007,2008 Jouni Hartikainen

    $Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $

    """

    # Default values for mean and covariance
    MM_def = np.zeros((dims, 1))
    PP_def = np.diag(np.ones(dims))

    # Number of models
    m = len(A)
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

    # Values of smoothed estimates at the last time step
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
    x_bki = np.empty(m, dtype=object)
    P_bki = np.empty(m, dtype=object)

    for i1 in range(m):
        x_bki[i1] = MM_def.copy()
        x_bki[i1][ind[i1]] = MM_i[-1, i1]
        P_bki[i1] = PP_def.copy()
        P_bki[i1][np.ix_(ind[i1], ind[i1])] = PP_i[-1, i1]

    # Space for model-conditioned backward_time predicted means and covariances
    x_kp = np.tile(MM_def, (m, 1, 1))
    P_kp = np.tile(PP_def, (m, 1, 1))

    for k in range(n - 2, -1, -1):
        a_j = np.zeros(m)
        mu_bijp = np.zeros((m, m))

        for i2 in range(m):
            a_j[i2] = np.sum(p_ijb[k][i2] * mu_bp)
            mu_bijp[:, i2] = p_ijb[k][:, i2] * mu_bp / a_j[i2]

            if type(A[i2]) == np.ndarray:
                A2 = A[i2]
            elif type(A[i2]) or callable(A[i2]):
                A2 = A[i2](x_bki[i2][ind[i2]], a_param[i2])
            else:
                A2 = A[i2](x_bki[i2][ind[i2]], a_param[i2])

            # Backward prediction
            #
            # Use KF is the dynamic model is linear
            if a is None or a[i2] is None:
                x_kp[i2][ind[i2]], P_kp[i2][np.ix_(ind[i2], ind[i2])] = (
                    KalmanFilter().predict(
                        x_bki[i2][ind[i2]],
                        P_bki[i2][np.ix_(ind[i2], ind[i2])],
                        linalg.inv(A2),
                        Q[i2],
                    )
                )
            else:
                x_kp[i2][ind[i2]], P_kp[i2][np.ix_(ind[i2], ind[i2])], _ = (
                    UnscentedKalmanFilter().predict(
                        x_bki[i2][ind[i2]],
                        P_bki[i2][np.ix_(ind[i2], ind[i2])],
                        a[i2],
                        Q[i2],
                        a_param[i2],
                    )
                )

        x_kp0 = np.tile(MM_def, (m, 1, 1))
        P_kp0 = np.tile(PP_def, (m, 1, 1))

        lhood_j = np.zeros(m)
        for i2 in range(m):
            P_kp0[i2][np.ix_(ind[i2], ind[i2])] = np.zeros((len(ind[i2]), len(ind[i2])))

            for i1 in range(m):
                x_kp0[i2][ind[i2]] += mu_bijp[i1, i2] * x_kp[i1][ind[i2]]

            for i1 in range(m):
                P_kp0[i2][np.ix_(ind[i2], ind[i2])] += mu_bijp[i1, i2] * (
                    P_kp[i1][np.ix_(ind[i2], ind[i2])]
                    + (x_kp[i1][ind[i2]] - x_kp0[i2][ind[i2]])
                    @ (x_kp[i1][ind[i2]] - x_kp0[i2][ind[i2]]).T
                )

            # Use KF if the measurement model is linear
            if h is None or h[i2] is None:
                (
                    x_bki[i2][ind[i2]],
                    P_bki[i2][np.ix_(ind[i2], ind[i2])],
                    _,
                    _,
                    _,
                    lhood_j[i2],
                ) = KalmanFilter().update(
                    x_kp0[i2][ind[i2]],
                    P_kp0[i2][np.ix_(ind[i2], ind[i2])],
                    Y[k],
                    H[i2],
                    R[i2],
                )
            else:
                (
                    x_bki[i2][ind[i2]],
                    P_bki[i2][np.ix_(ind[i2], ind[i2])],
                    _,
                    _,
                    _,
                    lhood_j[i2],
                ) = UnscentedKalmanFilter().update(
                    x_kp0[i2][ind[i2]],
                    P_kp0[i2][np.ix_(ind[i2], ind[i2])],
                    Y[k],
                    h[i2],
                    R[i2],
                    h_param[i2],
                )

        a_s = lhood_j @ a_j
        mu_bp = a_j * lhood_j / a_s

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

                lhood_ji[i2, i1], _ = stats.multivariate_normal.pdf(d_ijk, 0, D_ijk)

        d_j = np.zeros(m)
        for i2 in range(m):
            d_j[i2] = np.sum(p_ij[i2] * lhood_ji[i2])

        d = d_j @ MU[k]

        mu_ijsp = np.zeros((m, m))
        for i1 in range(m):
            for i2 in range(m):
                mu_ijsp[i1, i2] = p_ij[i2, i1] * lhood_ji[i2, i1] / d_j[i2]

        mu_sk[k] = d_j * MU[k] / d

        x_jis = np.empty((m, m), dtype=object)
        P_jis = np.empty((m, m), dtype=object)
        for i2 in range(m):
            for i1 in range(m):
                MM1 = MM_def.copy()
                MM1[ind[i2]] = MM_i[k, i2]

                PP1 = PP_def.copy()
                PP1[np.ix_(ind[i2], ind[i2])] = PP_i[k, i2]

                iPP1 = linalg.inv(PP1)
                iPP2 = linalg.inv(P_kp[i1])

                P_jis[i2, i1] = linalg.inv(iPP1 + iPP2)
                x_jis[i2, i1] = P_jis[i2, i1] @ (iPP1 @ MM1 + iPP2 @ x_kp[i1])

        for i2 in range(m):
            x_sik[k, i2] = MM_def.copy()
            P_sik[k, i2] = PP_def.copy()
            P_sik[k, i2][np.ix_(ind[i2], ind[i2])] = np.zeros(
                (len(ind[i2]), len(ind[i2]))
            )

            for i1 in range(m):
                x_sik[k, i2] += mu_ijsp[i1, i2] * x_jis[i2, i1]

            for i1 in range(m):
                P_sik[k, i2] += mu_ijsp[i1, i2] * (
                    P_jis[i2, i1]
                    + (x_jis[i2, i1] - x_sik[k, i2]) @ (x_jis[i2, i1] - x_sik[k, i2]).T
                )

        for i1 in range(m):
            x_sk[k] += mu_sk[k, i1] * x_sik[k, i1]

        for i1 in range(m):
            P_sk[k] += mu_sk[k, i1] * (
                P_sik[k, i1] + (x_sik[k, i1] - x_sk[k]) @ (x_sik[k, i1] - x_sk[k]).T
            )

    return x_sk, P_sk, x_sik, P_sik, mu_sk

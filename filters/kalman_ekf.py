import math
import sys
from copy import deepcopy
from typing import Callable, Union

import numpy as np
from numpy import linalg
from scipy import stats

from .helpers import pretty_str
from .transformers import reshape_z


class ExtendedKalmanFilter:
    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 0):
        """Initialize an object of Extended Kalman Filter

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
        SI :  numpy.array
            Inverse system uncertainty. Read only.
        log_likelihood : float
            log-likelihood of the last measurement. Read only.
        likelihood : float
            likelihood of last measurement. Read only.
        mahalanobis : float
            mahalanobis distance of the innovation. Read only.
        """

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

        # control transition matrix, used to apply control input to the state, initialized as None
        self.G = None
        # residual vector, difference between the measurement and the prediction, initialized to zero
        self.y = np.zeros((dim_z, 1))

        # measurement vector, reshaped to be consistent with the dimensionality of the state vector
        z = np.array([None] * self.dim_z)
        self.z = reshape_z(z, self.dim_z, self.x.ndim)

        # gain and residual are computed during the innovation step
        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros((dim_z, 1))
        # system uncertainty
        self.S = np.zeros((dim_z, dim_z))
        # inverse system uncertainty
        self.SI = np.zeros((dim_z, dim_z))
        # identity matrix.
        self._I = np.eye(dim_x)

        # only computed only if requested via property
        self._log_likelihood = math.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        # save the priors so that in case to inspect them for various purposes
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # save the posts so that in case to inspect them for various purposes
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_x(self, u: Union[np.array, float] = 0.0):
        """Predicts the next state of X. You would need to do this when the usual Taylor expansion to generate F is not providing accurate results.

        Parameters
        ----------
        u : Union[np.array, float], optional, by default 0.0
        """

        # x = Fx + Gu
        if self.G is not None and u is not None:
            self.x = self.F @ self.x + self.G @ u
        else:
            self.x = self.F @ self.x

    def predict(self, u: Union[np.array, float] = 0):
        """Predict next state (prior) using the Kalman filter state propagation equations.

        Parameters
        ----------
        u : Union[np.array, float], optional, by default 0
        """

        self.predict_x(u)
        self.P = self.F @ self.P @ self.F.T + self.Q

        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)

    def update(
        self,
        z: np.array,
        HJac: Callable[[np.array], np.ndarray],
        Hx: Callable[[np.array], np.ndarray],
        R: Union[np.array, float, None] = None,
        args: tuple = (),
        hx_args: tuple = (),
        residual=np.subtract,
    ):
        """Performs the update innovation of the EKF.

        Parameters
        ----------
        z : np.array
           measurement for this step.
        HJac : function
           function which computes the Jacobian of the H matrix. Takes state variable (self.x) as input, returns H.
        Hx : function
           function which takes as input the state variable (self.x) along with the optional arguments in hx_args, and returns the measurement that would correspond to that state.
        R : Union[np.array, float, None]
        args : tuple, optional, default (,)
           arguments to be passed into HJac after the required state variable.
        hx_args : tuple, optional, default (,)
           arguments to be passed into Hx function after the required state variable.
        residual : function (z, z2), optional
           function that computes the residual (difference) between the two measurement vectors.
        """

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        H = HJac(self.x, *args)
        hx = Hx(self.x, *hx_args)

        # update step
        Pxz = self.P @ H.T
        self.S = H @ Pxz + R
        self.SI = linalg.inv(self.S)
        self.K = Pxz @ self.SI

        self.y = residual(z, hx)
        self.x += self.K @ self.y

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update2(
        self,
        z: np.array,
        Hess: Callable[[np.array], np.ndarray],
        Hx: Callable[[np.array], np.ndarray],
        Hxx: Callable[[np.array], np.ndarray],
        HxR: Callable[[np.array], np.ndarray] = None,
        R: Union[np.array, float, None] = None,
        args: tuple = (),
        hx_args: tuple = (),
        hxx_args: tuple = (),
        hxr_args: tuple = (),
        residual=np.subtract,
    ):
        """Performs the update innovation of the 2nd order EKF.

        Parameters
        ----------
        z : np.array
            measurement for this step.
        Hess : function
            function which computes the Hessian of the H matrix. Takes state variable (self.x) as input, returns H.
        Hx : function
            function which takes as input the state variable (self.x) along with the optional arguments in hx_args, and returns the measurement that would correspond to that state.
        Hxx : function
            derivative of H matrix with respect to state as matrix.
        HxR : function
            derivative of H matrix with respect to noise as matrix, default identity.
        args : tuple, optional, default (,)
            arguments to be passed into Hess after the required state
            variable.
        hx_args : tuple, optional, default (,)
            arguments to be passed into Hx function after the required state variable.
        hxx_args : tuple, optional, default (,)
            arguments to be passed into Hxx function after the required state variable.
        hxr_args : tuple, optional, default (,)
            arguments to be passed into HxR function after the required state variable.
        residual : function (z, z2), optional
            function that computes the residual between the two measurement vectors. You will normally want to use the built in unless your residual computation is nonlinear (for example, angles)
        """
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()

        if not isinstance(args, tuple):
            args = (args,)
        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)
        if not isinstance(hxx_args, tuple):
            hxx_args = (hxx_args,)
        if not isinstance(hxr_args, tuple):
            hxr_args = (hxr_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        hxr = np.eye(self.dim_z) * R if HxR is None else HxR(self.x, *hxr_args)

        He = Hess(self.x, *args)
        hx = Hx(self.x, *hx_args)
        hxx = Hxx(self.x, *hxx_args)

        # update step
        self.y = residual(z, hx)
        for i in range(He.shape[0]):
            self.y[i] -= 0.5 * np.trace(He[i] @ self.P)

        self.S = hxr @ R @ hxr.T + hxx @ self.P @ hxx.T
        for i in range(He.shape[0]):
            for j in range(He.shape[0]):
                self.S[i, j] += 0.5 * np.trace(He[i] @ self.P @ He[j] @ self.P)

        self.S = hxx @ self.P @ hxx.T + R
        self.SI = linalg.inv(self.S)
        self.K = self.P @ hxx.T @ self.linalg.inv(self.S)
        self.x += self.K @ self.y

        I_KH = self._I - self.K @ hxx
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T

    def predict_update(
        self,
        z: np.array,
        HJac: Callable[[np.array], np.ndarray],
        Hx: Callable[[np.array], np.ndarray],
        R: Union[np.array, float, None] = None,
        args: tuple = (),
        hx_args: tuple = (),
        u: int = 0,
    ):
        """Performs the predict/update innovation of the EKF.

        Parameters
        ----------
        z : np.array
        HJac : Callable[[np.array], np.ndarray]
        Hx : Callable[[np.array], np.ndarray]
        R : Union[np.array, float, None]
        args : tuple
        hx_args : tuple
        u : int, optional, by default 0
        """

        # predict step
        self.predict(u)
        # update step
        self.update(z, HJac, Hx, R, args=args, hx_args=hx_args)

    @property
    def log_likelihood(self):
        """log-likelihood of the last measurement."""
        if self._log_likelihood is None:
            self._log_likelihood = stats.multivariate_normal.logpdf(
                x=self.y, cov=self.S, allow_singular=True
            )
        return self._log_likelihood

    @property
    def likelihood(self):
        """Computed from the log-likelihood. The log-likelihood can be very small, so by default we always return a number >= sys.float_info.min."""
        if self._likelihood is None:
            self._likelihood = np.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """Mahalanobis distance of innovation.
        e.g. 3 means measurement was 3 standard deviations away from the predicted value.
        """

        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(self.y.T @ self.SI @ self.y))
        return self._mahalanobis

    def __repr__(self):
        return "\n".join(
            [
                "KalmanFilter object",
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("S", self.S),
                pretty_str("likelihood", self.likelihood),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
            ]
        )


def etf_smooth1(
    M,
    P,
    Y,
    A=None,
    Q=None,
    ia=None,
    W=None,
    aparam=None,
    H=None,
    R=None,
    h=None,
    V=None,
    hparam=None,
    same_p_a=1,
    same_p_h=1,
):
    """
    ETF_SMOOTH1  Smoother based on two extended Kalman filters

    Syntax:
      [M,P] = ETF_SMOOTH1(M,P,Y,A,Q,ia,W,aparam,H,R,h,V,hparam,same_p_a,same_p_h)

    In:
      M - NxK matrix of K mean estimates from Kalman filter
      P - NxNxK matrix of K state covariances from Kalman Filter
      Y - Measurement vector
      A - Derivative of a() with respect to state as
          matrix, inline function, function handle or
          name of function in form A(x,param)       (optional, default eye())
      Q - Process noise of discrete model           (optional, default zero)
     ia - Inverse prediction function as vector,
          inline function, function handle or name
          of function in form ia(x,param)           (optional, default inv(A(x))*X)
      W - Derivative of a() with respect to noise q
          as matrix, inline function, function handle
          or name of function in form W(x,param)    (optional, default identity)
      aparam - Parameters of a. Parameters should be a single cell array, vector or a matrix
              containing the same parameters for each step or if different parameters
              are used on each step they must be a cell array of the format
              { param_1, param_2, ...}, where param_x contains the parameters for
              step x as a cell array, a vector or a matrix.   (optional, default empty)
      H  - Derivative of h() with respect to state as matrix,
           inline function, function handle or name
           of function in form H(x,param)
      R  - Measurement noise covariance.
      h  - Mean prediction (measurement model) as vector,
           inline function, function handle or name
           of function in form h(x,param).  (optional, default H(x)*X)
      V  - Derivative of h() with respect to noise as matrix,
           inline function, function handle or name
           of function in form V(x,param).  (optional, default identity)
      hparam - Parameters of h. See the description of aparam for the format of
                parameters.                  (optional, default aparam)
      same_p_a - If 1 uses the same parameters
                 on every time step for a    (optional, default 1)
      same_p_h - If 1 uses the same parameters
                 on every time step for h    (optional, default 1)

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

        fm, fP, *_ = ExtendedKalmanFilter().update(
            fm, fP, Y[k + 1], H, R, h, V, hparams
        )

        # Backward prediction
        if A is None:
            IA = None
        elif type(A) == np.ndarray:
            IA = linalg.inv(A)
        elif type(A) == str or callable(A):
            IA = linalg.inv(A(fm, aparams))
        else:
            IA = linalg.inv(A(fm, aparams))

        if W is None:
            B = np.eye(M.shape[1]) if Q is not None else np.eye(M.shape[1])

        elif type(W) == np.ndarray:
            B = W
        elif type(W) == str or callable(W):
            B = W(fm, aparams)
        else:
            B = W(fm, aparams)

        IQ = IA @ B @ Q @ B.T @ IA.T

        fm, fP = ExtendedKalmanFilter().predict(fm, fP, IA, IQ, ia, None, aparams)

        BM[k] = fm
        BP[k] = fP

    # Combine estimates
    for k in range(m_1 - 1):
        tmp = linalg.inv(linalg.inv(P[k]) + linalg.inv(BP[k]))
        M[k] = tmp @ (linalg.solve(P[k], M[k]) + linalg.solve(BP[k], BM[k]))
        P[k] = tmp

    return M, P


def erts_smooth1(M, P, A=None, Q=None, a=None, W=None, param=None, same_p=True):
    """
    ERTS_SMOOTH1  Extended Rauch-Tung-Striebel smoother

    Syntax:
      [M,P,D] = ERTS_SMOOTH1(M,P,A,Q,[a,W,param,same_p])

    In:
      M - NxK matrix of K mean estimates from Unscented Kalman filter
      P - NxNxK matrix of K state covariances from Unscented Kalman Filter
      A - Derivative of a() with respect to state as
          matrix, inline function, function handle or
          name of function in form A(x,param)                 (optional, default eye())
      Q - Process noise of discrete model                       (optional, default zero)
      a - Mean prediction E[a(x[k-1],q=0)] as vector,
          inline function, function handle or name
          of function in form a(x,param)                        (optional, default A(x)*X)
      W - Derivative of a() with respect to noise q
          as matrix, inline function, function handle
          or name of function in form W(x,param)                (optional, default identity)
      param - Parameters of a. Parameters should be a single cell array, vector or a matrix
              containing the same parameters for each step or if different parameters
              are used on each step they must be a cell array of the format
              { param_1, param_2, ...}, where param_x contains the parameters for
              step x as a cell array, a vector or a matrix.     (optional, default empty)
      same_p - 1 if the same parameters should be
               used on every time step                          (optional, default 1)

    Out:
      K - Smoothed state mean sequence
      P - Smoothed state covariance sequence
      D - Smoother gain sequence

    Description:
      Extended Rauch-Tung-Striebel smoother algorithm. Calculate
      "smoothed" sequence from given Kalman filter output sequence by
      conditioning all steps to all measurements.
    """

    m_1, m_2 = M.shape[:2]

    # Apply defaults

    if A is None:
        A = np.eye(m_2)

    if Q is None:
        Q = np.zeros(m_2)

    if W is None:
        W = np.eye(m_2)

    # Extend Q if NxN matrix
    if len(Q.shape) < 3:
        Q = np.tile(Q, (m_1, 1, 1))

    # Run the smoother
    M = M.copy()
    P = P.copy()
    D = np.zeros((m_1, m_2, m_2))
    for k in range(m_1 - 2, -1, -1):
        if param is None:
            params = None
        elif same_p:
            params = param
        else:
            params = param[k]

        # Perform prediction
        if a is None:
            m_pred = A @ M[k]
        elif type(a) == np.ndarray:
            m_pred = a
        elif type(a) == str or callable(a):
            m_pred = a(M[k], params)
        else:
            m_pred = a(M[k], params)

        if type(A) == np.ndarray:
            F = A
        elif type(A) or callable(A):
            F = A(M[k], params)
        else:
            F = A(M[k], params)

        if type(W) == np.ndarray:
            B = W
        elif type(W) or callable(W):
            B = W(M[k], params)
        else:
            B = W(M[k], params)

        P_pred = F @ P[k] @ F.T + B @ Q[k] @ B.T
        C = P[k] @ F.T

        D[k] = linalg.solve(P_pred.T, C.T).T
        M[k] += D[k] @ (M[k + 1] - m_pred)
        P[k] += D[k] @ (P[k + 1] - P_pred) @ D[k].T

    return M, P, D


def eimm_predict(X_ip, P_ip, MU_ip, p_ij, ind, dims, A, a, param, Q, nargout=3):
    """
    IMM_PREDICT  Interacting Multiple Model (IMM) Filter prediction step

    Syntax:
      [X_p,P_p,c_j,X,P] = EIMM_PREDICT(X_ip,P_ip,MU_ip,p_ij,ind,dims,A,a,param,Q)

    In:
      X_ip  - Cell array containing N^j x 1 mean state estimate vector for
              each model j after update step of previous time step
      P_ip  - Cell array containing N^j x N^j state covariance matrix for
              each model j after update step of previous time step
      MU_ip - Vector containing the model probabilities at previous time step
      p_ij  - Model transition matrix
      ind   - Indices of state components for each model as a cell array
      dims  - Total number of different state components in the combined system
      A     - Dynamic model matrices for each linear model and Jacobians of each
              non-linear model's measurement model function as a cell array
      a     - Function handles of dynamic model functions for each model
              as a cell array
      param - Parameters of a for each model as a cell array
      Q     - Process noise matrices for each model as a cell array.

    Out:
      X_p  - Predicted state mean for each model as a cell array
      P_p  - Predicted state covariance for each model as a cell array
      c_j  - Normalizing factors for mixing probabilities
      X    - Combined predicted state mean estimate
      P    - Combined predicted state covariance estimate

    Description:
      IMM-EKF filter prediction step. If some of the models have linear
      dynamics standard Kalman filter prediction step is used for those.

    See also:
      EIMM_UPDATE, EIMM_SMOOTH

    History:
      09.01.2008 JH The first official version.

    Copyright (C) 2007,2008 Jouni Hartikainen

    $Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $

    """

    # Number of models
    m = len(X_ip)

    # Construct empty cell arrays for ekf_update if a is not specified
    if a is None:
        a = np.empty(m, dtype=object)

        # Same for a's parameters
    if param is None:
        param = np.empty(m, dtype=object)

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
        X_p[i], P_p[i] = ExtendedKalmanFilter().predict(
            X_0j[i][ind[i]],
            P_0j[i][np.ix_(ind[i], ind[i])],
            A[i],
            Q[i],
            a[i],
            None,
            param[i],
        )
        # [X_p{i}, P_p{i}] = kf_predict(X_0j{i}(ind{i}),P_0j{i}(ind{i},ind{i}),A{i},Q{i})

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


def eimm_update(X_p, P_p, c_j, ind, dims, Y, H, h, R, param, nargout=5):
    """
    IMM_UPDATE  Interacting Multiple Model (IMM) Filter update step

    Syntax:
      [X_i,P_i,MU,X,P] = IMM_UPDATE(X_p,P_p,c_j,ind,dims,Y,H,h,R,param)

    In:
      X_p  - Cell array containing N^j x 1 mean state estimate vector for
             each model j after prediction step
      P_p  - Cell array containing N^j x N^j state covariance matrix for
             each model j after prediction step
      c_j  - Normalizing factors for mixing probabilities
      ind  - Indices of state components for each model as a cell array
      dims - Total number of different state components in the combined system
      Y    - Dx1 measurement vector.
      H    - Measurement matrices for each linear model and Jacobians of each
             non-linear model's measurement model function as a cell array
      h    - Cell array containing function handles for measurement functions
             for each model having non-linear measurements
      R    - Measurement noise covariances for each model as a cell array.
      param - Parameters of h

    Out:
      X_i  - Updated state mean estimate for each model as a cell array
      P_i  - Updated state covariance estimate for each model as a cell array
      MU   - Estimated probabilities of each model
      X    - Combined updated state mean estimate
      P    - Combined updated covariance estimate

    Description:
      IMM-EKF filter measurement update step. If some of the models have linear
      measurements standard Kalman filter update step is used for those.

    See also:
      IMM_PREDICT, IMM_SMOOTH, IMM_FILTER

    History:
      01.11.2007 JH The first official version.



    $Id: imm_update.m 111 2007-11-01 12:09:23Z jmjharti $

    """

    # Number of models
    m = len(X_p)

    # Construct empty cell arrays for ekf_update if h is not specified
    if h is None:
        h = np.empty(m, dtype=object)

    # Same for h's parameters
    if param is None:
        param = np.empty(m, dtype=object)

    # Space for update state mean, covariance and likelihood of measurements
    X_i = np.empty(m, dtype=object)
    P_i = np.empty(m, dtype=object)
    lbda = np.zeros(m)

    # Update for each model
    for i in range(m):
        # Update the state estimates
        # [X_i{i}, P_i{i}, K, IM, IS, lbda(i)] = ekf_update1(X_p{i},P_p{i},Y,H{i},R{i},h{i},[],param{i})
        X_i[i], P_i[i], _, _, _, lbda[i] = ExtendedKalmanFilter().update(
            X_p[i], P_p[i], Y, H[i], R[i], h[i], None, param[i], nargout=6
        )
        # [X_i{i}, P_i{i}, K, IM, IS] = kf_update(X_p{i},P_p{i},Y,H{i},R{i})

        # Calculate measurement likelihoods for each model
        # lbda(i) = kf_lhood(X_p{i},P_p{i},Y,H{i},R{i})

    # Calculate the model probabilities
    MU = np.zeros(m)
    c = np.sum(lbda * c_j)
    MU = c_j * lbda / c

    # In case lbda's happen to be zero
    # if c == 0
    if c == 0:
        MU = c_j
        # X_i = X_p
        # P_i = P_p

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


def eimm_smooth(
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
    EIMM_SMOOTH  EKF based fixed-interval IMM smoother using two IMM-EKF filters.

    Syntax:
      [X_S,P_S,X_IS,P_IS,MU_S] = EIMM_SMOOTH(MM,PP,MM_i,PP_i,MU,p_ij,mu_0j,ind,dims,A,a,a_param,Q,R,H,h,h_param,Y)

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
      EKF based two-filter fixed-interval IMM smoother.

    See also:
      EIMM_UPDATE, EIMM_PREDICT

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
    x_bki = np.empty(m, dtype=object)
    P_bki = np.empty(m, dtype=object)

    # Initialize with default values
    for i1 in range(m):
        x_bki[i1] = MM_def.copy()
        x_bki[i1][ind[i1]] = MM_i[-1, i1]
        P_bki[i1] = PP_def.copy()
        P_bki[i1][np.ix_(ind[i1], ind[i1])] = PP_i[-1, i1]

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

            # Retrieve the transition matrix or the Jacobian of the dynamic model
            if type(A[i2]) == np.ndarray:
                A2 = A[i2]
            elif type(A[i2]) == str or callable(A[i2]):
                A2 = A[i2](x_bki[i2][ind[i2]], a_param[i2])
            else:
                A2 = A[i2](x_bki[i2][ind[i2]], a_param[i2])

            # Backward-time EKF prediction step
            x_kp[i2][ind[i2]], P_kp[i2][np.ix_(ind[i2], ind[i2])] = (
                ExtendedKalmanFilter().predict(
                    x_bki[i2][ind[i2]],
                    P_bki[i2][np.ix_(ind[i2], ind[i2])],
                    linalg.inv(A2),
                    Q[i2],
                    a[i2],
                    None,
                    a_param[i2],
                )
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

            # Backward-time EKF update
            #
            # If the measurement model is linear don't pass h and h_param to ekf_update1
            if h is None or h[i2] is None:
                (
                    x_bki[i2][ind[i2]],
                    P_bki[i2][np.ix_(ind[i2], ind[i2])],
                    _,
                    _,
                    _,
                    lhood_j[i2],
                ) = ExtendedKalmanFilter().update(
                    x_kp0[i2][ind[i2]],
                    P_kp0[i2][np.ix_(ind[i2], ind[i2])],
                    Y[k],
                    H[i2],
                    R[i2],
                    None,
                    None,
                    None,
                    nargout=6,
                )
            else:
                (
                    x_bki[i2][ind[i2]],
                    P_bki[i2][np.ix_(ind[i2], ind[i2])],
                    _,
                    _,
                    _,
                    lhood_j[i2],
                ) = ExtendedKalmanFilter().predict(
                    x_kp0[i2][ind[i2]],
                    P_kp0[i2][np.ix_(ind[i2], ind[i2])],
                    Y[k],
                    H[i2],
                    R[i2],
                    h[i2],
                    None,
                    h_param[i2],
                    nargout=6,
                )

        # Normalizing constant
        a_s = lhood_j @ a_j
        # Updated model probabilities
        mu_bp = a_j * lhood_j / a_s

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

                iPP1 = linalg.inv(PP1)
                iPP2 = linalg.inv(P_kp[i1])

                # Covariance of the Gaussian product
                P_jis[i2, i1] = linalg.inv(iPP1 + iPP2)
                # Mean of the Gaussian product
                x_jis[i2, i1] = P_jis[i2, i1] @ (iPP1 @ MM1 + iPP2 @ x_kp[i1])

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

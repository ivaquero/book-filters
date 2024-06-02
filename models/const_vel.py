import sys

import numpy as np
from scipy import linalg

from .noise import white_noise_discrete
from .ssmodel_linear import LinearStateSpaceModel

sys.path.append("..")
from filters.kalman import KalmanFilter


class ConstantVelocity(LinearStateSpaceModel):
    """Linear constant velocity model.

    State space model of the plant.
    x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t]
    z[t] = H[t]*x[t] + M[t]*v[t]

    x: state, [x1, vx1, x2, vx2]
    z: output, [x1, x2]
    u: control input
    w: system noise
    v: observation noise
    """

    # # demensions
    NDIM = {
        "x": 4,  # state
        "z": 2,  # measurement
        "u": 0,  # control input
        "w": 2,  # process noise
        "v": 2,  # observation noise
    }

    def __init__(self, dt=0.1):
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.L = np.array([[0.5 * (dt**2), 0], [dt, 0], [0, 0.5 * (dt**2)], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.M = np.eye(2)

    def state_equation(self, t, x, u=0, w=np.zeros(2)):
        """Sate equation.

        x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t],
        x: state, [x1, vx1, x2, vx2]
        """
        return self.F @ x + self.L @ w

    def observation_equation(self, t, x, v=np.zeros(2)):
        """Observation equation.

        z[t] = H[t]*x[t] + M[t]*v[t],
        x: state, [x1, vx1, x2, vx2]
        z: output, [x1, x2]
        """
        return self.H @ x + self.M @ v

    def Ft(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return F[t].
        """
        return self.F

    def Lt(self, t):
        """x[t+1] = F[t]*x[t] + G[t]*u[t] + L[t]*w[t].

        return L[t].
        """
        return self.L

    def Ht(self, t):
        """z[t] = H[t]*x[t] + M[t]*v[t].

        return H[t].
        """
        return self.H

    def Mt(self, t):
        """z[t] = H[t]*x[t] + M[t]*v[t].

        return M[t] .
        """
        return self.M


def H(dim_x, dim_z):
    if dim_z == 1:
        H = np.atleast_2d([1] + [0] * (dim_x - 1))
    elif dim_z == 2:
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    elif dim_z > 2:
        H = np.eye(dim_x)
    else:
        ValueError("dim_z >= 1")
    return H


def FCV(dim, dt):
    """State transition matrix for a constant velocity model"""
    F = np.array([[1, dt], [0, 1]], dtype=float)
    # [x, x']'
    if dim == 2:
        F = F
    # [x, x', y, y']'
    elif dim == 4:
        F = linalg.block_diag(F, F)
    # [x, x', y, y', z, z']'
    elif dim == 6:
        F = linalg.block_diag(F, F, F)
    else:
        ValueError("dim must be 2, 4, 6")
    return F


def FxCV(x, dt):
    """Shortcut function for a constant velocity model"""
    return FCV(len(x), dt) @ x


def KFCV1d(P, R, Q=0, dt=1, x=[0]):
    if type(x) == list:
        x = np.array(x)
    dim_x = len(x)
    kf_cv = KalmanFilter(dim_x=dim_x, dim_z=1)
    kf_cv.x = np.zeros(dim_x)
    kf_cv.F = np.eye(1)
    kf_cv.H = np.eye(1)
    kf_cv.R *= R
    kf_cv.Q *= Q
    kf_cv.P *= P

    return kf_cv


def KFCV2d(P, R, Q=0, dt=1, x=[0, 0]):
    if type(x) == list:
        x = np.array(x)
    dim_x = len(x)
    kf_cv = KalmanFilter(dim_x=dim_x, dim_z=1)
    kf_cv.x = np.zeros(dim_x)
    kf_cv.F = FCV(dim_x, dt)
    kf_cv.H = H(dim_x, dim_z=1)
    kf_cv.R *= R

    if np.isscalar(P):
        kf_cv.P *= P
    else:
        kf_cv.P = np.atleast_2d(P)

    if np.isscalar(Q):
        kf_cv.Q = white_noise_discrete(dim=dim_x, dt=dt, var=Q)
    else:
        kf_cv.Q = np.atleast_2d(Q)
    return kf_cv

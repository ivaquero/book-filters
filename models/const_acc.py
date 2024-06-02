import sys

import numpy as np
from scipy import linalg

from .noise import white_noise_discrete

sys.path.append("..")
from filters.kalman import KalmanFilter


def H(dim_x, dim_z):
    if dim_z == 1:
        H = np.atleast_2d([1] + [0] * (dim_x - 1))
    elif dim_z > 1:
        H = np.eye(dim_x)
    else:
        ValueError("dim_z >= 1")
    return H


def FxCA(x, dt):
    """Shortcut function for a constant acceleration model"""
    return FCA(len(x), dt) @ x


def FCA(dim, dt):
    """State transition matrix for a constant acceleration model"""
    F = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]], dtype=float)
    # [x, x', x'']'
    if dim == 3:
        F = F
    # [x, x', x'', y, y', y'']'
    elif dim == 6:
        F = linalg.block_diag(F, F)
    # [x, x', x'', y, y', y'', z, z', z'']'
    elif dim == 9:
        F = linalg.block_diag(F, F, F)
    else:
        ValueError("dim must be 3, 6, 9")
    return F


def KFCA3d(P, R, Q=0, dt=1, x=[0, 0, 0]):
    if type(x) == list:
        x = np.array(x)
    dim_x = len(x)
    kf_ca = KalmanFilter(dim_x=dim_x, dim_z=1)
    kf_ca.x = x
    kf_ca.F = FCA(dim_x, dt)
    kf_ca.H = H(dim_x, dim_z=1)
    kf_ca.R *= R
    if np.isscalar(P):
        kf_ca.P *= P
    else:
        kf_ca.P = np.atleast_2d(P)
    if np.isscalar(Q):
        kf_ca.Q = white_noise_discrete(dim=dim_x, dt=dt, var=Q)
    else:
        kf_ca.Q = np.atleast_2d(Q)
    return kf_ca

import sys

import numpy as np
from numpy import random

from .helpers import KFSaver
from .kalman import KalmanFilter
from .kalman_ukf import UnscentedKalmanFilter
from .sigma_points import JulierSigmas

sys.path.append("..")
from models.constant_velocity import FCV


def fusion_kf2d(sensor1_sigma, sensor2_sigma, dt=0.1, P=100, seed=1123):
    kf = KalmanFilter(dim_x=2, dim_z=2)
    kf.x = np.array([[0.0], [1.0]])
    kf.F = FCV(2, dt)
    kf.H = np.array([[1.0, 0.0], [1.0, 0.0]])
    kf.P *= P
    kf.Q *= np.array([[(dt**3) / 3, (dt**2) / 2], [(dt**2) / 2, dt]]) * 0.02
    kf.R = [[sensor1_sigma**2, 0], [0, sensor2_sigma**2]]
    saver = KFSaver(kf)

    random.seed(seed)
    for i in range(1, 100):
        m0 = i + random.randn() * sensor1_sigma
        m1 = i + random.randn() * sensor2_sigma
        kf.predict()
        kf.update(np.array([[m0], [m1]]))
        saver.save()
    saver.to_array()
    print(f"fusion std: {np.std(saver.y[:, 0]):.3f}")

    return saver


def fusion_ukf2d(hx, fx):
    # create unscented Kalman filter with large initial uncertainty
    points = JulierSigmas(2, kappa=2)
    ukf = UnscentedKalmanFilter(2, 2, 0.1, hx=hx, fx=fx, points=points)
    ukf.x = np.array([100, 100])
    ukf.P *= 40
    return ukf

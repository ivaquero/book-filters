from collections import namedtuple

import numpy as np
from numpy import linalg

gaussian = namedtuple("Gaussian", ["mean", "var"])  # noqa: PYI024
gaussian.__repr__ = lambda s: f"N(μ={s[0]:.3f}, σ²={s[1]**2:.3f})"


def gaussian_product(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


def gaussian_sum(g1, g2):
    return gaussian(g1.mean + g2.mean, g1.var + g2.var)


def NEES(xs, est_xs, ps):
    est_err = xs - est_xs
    return [x.T @ linalg.inv(p) @ x for x, p in zip(est_err, ps)]


def multi_gaussian_product(mean1, cov1, mean2, cov2):
    cov1 = np.asarray(cov1)
    cov2 = np.asarray(cov2)
    mean1 = np.asarray(mean1)
    mean2 = np.asarray(mean2)

    sum_inv = np.linalg.inv(cov1 + cov2)
    mean = cov2 @ sum_inv @ mean1 + cov1 @ sum_inv @ mean2
    cov = cov1 @ sum_inv @ cov2

    return mean, cov

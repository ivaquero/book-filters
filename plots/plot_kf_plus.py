import sys

import numpy as np
from numpy import random
from scipy import stats

sys.path.append("..")
from filters.kalman_ukf import unscented_transform
from filters.sigma_points import MerweScaledSigmas


def plot_ekf_vs_mc(ax):
    def fx(x):
        return x**3

    def dfx(x):
        return 3 * x**2

    mean = 1
    var = 0.1
    std = np.sqrt(var)

    data = random.normal(loc=mean, scale=std, size=50000)
    d_t = fx(data)

    ekf_mean = fx(mean)
    slope = dfx(mean)
    ekf_std = abs(slope * std)

    norm = stats.norm(ekf_mean, ekf_std)
    xs = np.linspace(-3, 5, 200)
    ax.plot(xs, norm.pdf(xs), lw=2, ls="--", color="b")
    ax.hist(
        d_t,
        bins=200,
        density=True,
        histtype="step",
        lw=2,
        color="g",
    )

    actual_mean = d_t.mean()
    ax.axvline(actual_mean, lw=2, color="g", label="Monte Carlo")
    ax.axvline(ekf_mean, lw=2, ls="--", color="b", label="EKF")
    ax.legend()

    print(f"actual mean={d_t.mean():.2f}, std={d_t.std():.2f}")
    print(f"EKF    mean={ekf_mean:.2f}, std={ekf_std:.2f}")


def plot_ukf_vs_mc(ax, kappa=1.0, alpha=0.001, beta=3.0):
    def fx(x):
        return x**3

    def dfx(x):
        return 3 * x**2

    mean = 1
    var = 0.1
    std = np.sqrt(var)

    data = random.normal(loc=mean, scale=std, size=50000)
    d_t = fx(data)

    points = MerweScaledSigmas(1, kappa, alpha, beta)
    Wm, Wc = points.Wm, points.Wc
    sigmas = points.sigma_points(mean, var)

    sigmas_f = np.zeros((3, 1))
    for i in range(3):
        sigmas_f[i] = fx(sigmas[i, 0])

    ### pass through unscented transform
    ukf_mean, ukf_cov = unscented_transform(sigmas_f, Wm, Wc)
    ukf_mean = ukf_mean[0]
    ukf_std = np.sqrt(ukf_cov[0])

    norm = stats.norm(ukf_mean, ukf_std)
    xs = np.linspace(-3, 5, 200)
    ax.plot(xs, norm.pdf(xs), ls="--", lw=2, color="b")
    ax.hist(
        d_t,
        bins=200,
        density=True,
        histtype="step",
        lw=2,
        color="g",
    )

    actual_mean = d_t.mean()
    ax.axvline(actual_mean, lw=2, color="g", label="Monte Carlo")
    ax.axvline(ukf_mean, lw=2, ls="--", color="b", label="UKF")
    ax.legend()

    print(f"actual mean={d_t.mean():.2f}, std={d_t.std():.2f}")
    print(f"UKF    mean={ukf_mean:.2f}, std={ukf_std[0]:.2f}")


def show_linearization(ax, tan_x=1.5):
    xs = np.arange(0, 2, 0.01)
    ys = [x**2 - 2 * x for x in xs]

    tan_y = tan_x**2 - 2 * tan_x

    def tan_l(x):
        return (2 * tan_x - 2) * (x - tan_x) + tan_y

    ax.plot(xs, ys, label="$f(x)=x^2−2x$")
    ax.plot(
        [tan_x - 0.5, tan_x + 0.5],
        [tan_l(tan_x - 0.5), tan_l(tan_x + 0.5)],
        color="k",
        ls="--",
        label="linearization",
    )
    ax.axvline(tan_x, lw=1, c="k")
    ax.set(
        xlabel=f"$x={tan_x}$",
        title=f"Linearization of $f(x)$ at $x={tan_x}$",
    )
    ax.legend()

import sys

import numpy as np

from .plot_common import plot_cov_ellipse

sys.path.append("..")
from filters.sigma_points import MerweScaledSigmas


def plot_sigmas(ax, sigmas, x, cov):
    if not np.isscalar(cov):
        cov = np.atleast_2d(cov)
    pts = sigmas.sigma_points(x=x, P=cov)
    ax.scatter(pts[:, 0], pts[:, 1], s=sigmas.Wm * 1000)
    ax.axis("equal")
    ax.grid(True, linestyle="--")


def _plot_sigmas(ax, s, w, alpha=0.5, **kwargs):
    min_w = min(abs(w))
    scale_factor = 100 / min_w
    ax.scatter(
        s[:, 0],
        s[:, 1],
        s=abs(w) * scale_factor,
        alpha=alpha,
        **kwargs,
    )


def plot_sigmas_selection(ax, kappas=None, alphas=None, betas=None, var=None):
    if kappas is None:
        kappas = [1.0, 0.15, 10]
    if alphas is None:
        alphas = [0.09, 0.15, 0.2]
    if betas is None:
        betas = [2.0, 1.0, 3.0]
    if var is None:
        var = [0.5]
    P = np.array([[3, 1.1], [1.1, 4]])

    xs = np.array([[2, 5], [5, 5], [8, 5]])

    for x, κ, α, β in zip(xs, kappas, alphas, betas):
        points = MerweScaledSigmas(2, κ, α, β)
        sigmas = points.sigma_points(x, P)
        _plot_sigmas(ax, sigmas, points.Wc, alpha=1.0, facecolor="k")
        plot_cov_ellipse(
            ax,
            x,
            P,
            stds=np.sqrt(var),
            facecolor="b",
            alpha=0.3,
            title=False,
        )

    ax.axis("equal")
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def plot_sigmas_compar_param(axes, obj, kappas=None, alphas=None, betas=None, var=None):
    if kappas is None:
        kappas = [1.0, 1.0]
    if alphas is None:
        alphas = [0.3, 1.0]
    if betas is None:
        betas = [2.0, 2.0]
    if var is None:
        var = [1, 4]
    x = np.array([0, 0])
    P = np.array([[4, 2], [2, 4]])

    for ax, κ, α, β in zip(axes, kappas, alphas, betas):
        sigmas = MerweScaledSigmas(n=2, kappa=κ, alpha=α, beta=β)
        _plot_sigmas(ax, sigmas.sigma_points(x, P), sigmas.Wc, c="b")
        plot_cov_ellipse(ax, x, P, stds=np.sqrt(var), facecolor="g", alpha=0.2)

        obj_dict = {"kappa": κ, "alpha": α, "beta": β}
        ax.set(title=f"{obj} = {obj_dict[obj]}")

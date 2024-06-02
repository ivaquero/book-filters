import math

import numpy as np
from matplotlib import patches


def prepend_x0(x0, data):
    if isinstance(x0, list):
        data = [x0] + data.tolist()
    if isinstance(x0, np.ndarray):
        data = np.concatenate([x0], data)
    return data


def gen_data_by_only_x(xs, dt):
    return np.arange(0, len(xs) * dt, dt), xs


def plot_zs(
    ax,
    xs,
    ys=None,
    x0=None,
    dt=1,
    label="Measured",
    **scatter_kwargs,
):
    if x0:
        xs = prepend_x0(x0, xs)
    if ys is None:
        xs, ys = gen_data_by_only_x(xs, dt)

    ax.scatter(
        xs,
        ys,
        label=label,
        marker="x",
        color="k",
        **scatter_kwargs,
    )
    # ax.set(xlim=[-1, len(xs) + 1], ylim=[-1, len(ys) + 1])
    ax.grid(1)


def plot_track(ax, xs, ys=None, dt=None, label="Track", c="k", lw=2, ls=":", **kwargs):
    if ys is None and dt is not None:
        xs, ys = gen_data_by_only_x(xs, dt)
    if ys is not None:
        return ax.plot(xs, ys, color=c, lw=lw, ls=ls, label=label, **kwargs)

    return ax.plot(xs, color=c, lw=lw, ls=ls, label=label, **kwargs)


def plot_preds(ax, priors, kind=None, scatter=False):
    rng = range(len(priors))
    if kind == "scatter":
        ax.scatter(rng, priors, marker="d", label="Predicted", color="r")
    else:
        ax.plot(rng, priors, ls="-.", label="Predicted", color="r")
    ax.legend()


def plot_cov2d(axes, cov):
    axes[0].set(title=r"$σ^2_x$")
    plot_covariance(axes[0], cov, (0, 0))
    axes[1].set(title=r"$σ^2_ẋ$")
    plot_covariance(axes[1], cov, (1, 1))


def plot_covariance(ax, P, index=(0, 0)):
    ps = [p[index[0], index[1]] for p in P]
    ax.plot(ps)


def plot_track_ellipses(ax, N, zs, xs, cov, title):
    for i, p in enumerate(cov):
        plot_cov_ellipse(
            ax,
            (i + 1, xs[i]),
            cov=p,
            stds=[2] * N,
            edgecolor="darkslateblue",
            facecolor="white",
        )

    ax.set(title=title)
    return p


def cal_cov_ellipse(cov, deviations=1):
    U, s, _ = np.linalg.svd(cov)
    orientation = math.atan2(U[1, 0], U[0, 0])
    width_radius = deviations * np.sqrt(s[0])
    height_radius = deviations * np.sqrt(s[1])

    if height_radius > width_radius:
        raise ValueError("width_radius must be greater than height_radius")

    return (orientation, width_radius, height_radius)


def plot_cov_ellipse(
    ax,
    mean,
    cov,
    stds=None,
    show_semiaxis=False,
    show_center=True,
    angle=1,
    edgecolor="darkslateblue",
    facecolor="green",
    alpha=0.2,
    label="",
    title=True,
    **line_kwargs,
):
    if stds is None:
        stds = [1]
    ellipse = cal_cov_ellipse(cov)

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.0
    height = ellipse[2] * 2.0

    for stdi in stds:
        e = patches.Ellipse(mean, stdi * width, stdi * height, angle=angle)
        ax.add_patch(e)
        e.set(
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            label=label,
            **line_kwargs,
        )

    x, y = mean
    if show_center:
        ax.scatter(x, y, marker="+", color=edgecolor)
    if show_semiaxis:
        a = ellipse[0]
        h, w = height / 4, width / 4
        ax.plot(
            [x, x + h * math.cos(a + math.pi / 2)],
            [y, y + h * math.sin(a + math.pi / 2)],
        )
        ax.plot([x, x + w * math.cos(a)], [y, y + w * math.sin(a)])
    if title:
        ax.set(title=f"[{cov[0]}\n   {cov[1]}]")


def plot_resids_lims(ax, Ps, stds=1.0):
    std = np.sqrt(Ps) * stds

    ax.plot(-std, color="k", ls=":", lw=2)
    ax.plot(std, color="k", ls=":", lw=2)
    ax.fill_between(range(len(std)), -std, std, facecolor="#ffff00", alpha=0.3)


def plot_resids(ax, xs, data, col, ylabel, title=None, limits=True, stds=1):
    res = xs - data.x[:, col]
    ax.plot(res)
    if limits:
        Ps = data.P[:, col, col]
        plot_resids_lims(ax, Ps, stds=stds)
    ax.set(title=title, xlabel="time", ylabel=ylabel)

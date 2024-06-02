import numpy as np

from .plot_common import gen_data_by_only_x, plot_track, plot_zs


def plot_kf(
    ax,
    xs,
    ys=None,
    dt=1,
    var=None,
    label="Filter",
    band_color="blue",
    **kwargs,
):
    if ys is None:
        xs, ys = gen_data_by_only_x(xs, dt)
    ax.plot(xs, ys, label=label, **kwargs)

    if var is not None:
        std = np.sqrt(var)
        ax.plot(xs, ys + std, linestyle=":", color="k", lw=2)
        ax.plot(xs, ys - std, linestyle=":", color="k", lw=2)
        ax.fill_between(
            xs,
            ys - std,
            ys + std,
            facecolor=band_color,
            alpha=0.1,
        )


def plot_kf_track(ax, xs, filter_xs, zs, label=None, title=None):
    plot_kf(ax, filter_xs[:, 0])
    plot_track(ax, xs[:, 0])

    if zs is not None:
        plot_zs(ax, zs, label=label)

    ax.set(
        title=title,
        xlabel="time",
        ylabel="meters",
        xlim=(-1, len(xs)),
    )
    ax.legend()


def plot_kf_with_cov(
    ax,
    xs,
    cov,
    track,
    zs,
    std_scale=1,
    y_lim=None,
    xlabel="time",
    ylabel="position",
    title="Kalman Filter",
):
    num = len(zs)
    zs = np.asarray(zs)

    cov = np.asarray(cov)
    std = std_scale * np.sqrt(cov[:, 0, 0])
    std_top = np.minimum(track + std, [num + 10])
    std_btm = np.maximum(track - std, [-50])

    std_top = track + std
    std_btm = track - std

    plot_track(ax, track, c="k")
    plot_zs(ax, xs=zs)
    plot_kf(ax, xs)

    ax.plot(std_top, linestyle=":", color="k", lw=1, alpha=0.4)
    ax.plot(std_btm, linestyle=":", color="k", lw=1, alpha=0.4)
    ax.fill_between(
        range(len(std_top)),
        std_top,
        std_btm,
        facecolor="yellow",
        alpha=0.2,
        interpolate=True,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, num), title=title)
    if y_lim is not None:
        ax.set(ylim=y_lim)
    else:
        ax.set(ylim=(-50, num + 10))
    ax.legend()


def plot_kf_with_resids(axes, dt, xs, z_xs, res):
    t = np.arange(0, len(xs) * dt, dt)
    if z_xs is not None:
        plot_zs(axes[0], xs=t, ys=z_xs, dt=dt, label="z")
    plot_kf(axes[0], xs=t, ys=xs, dt=dt)
    axes[0].set(
        xlabel="time",
        ylabel="X",
        title="estimates vs measurements",
    )
    axes[0].legend()

    axes[1].plot(t, res)
    axes[1].set(xlabel="time", ylabel="residual", title="residuals")

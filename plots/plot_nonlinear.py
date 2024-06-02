import matplotlib.pyplot as ax
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_transferred_gaussian(
    data,
    func,
    func_name="f(x)",
    out_lim=None,
    num_bins=300,
    figsize=(8, 6),
):
    ys = func(data)
    std = np.std(ys)

    x0 = np.mean(data)
    in_std = np.std(data)
    in_lims = [x0 - in_std * 3, x0 + in_std * 3]

    y = func(x0)
    if out_lim is None:
        out_lim = [y - std * 3, y + std * 3]

    _, axes = ax.subplots(2, 2, figsize=figsize, constrained_layout=True)

    # plot output
    h = np.histogram(ys, num_bins, density=False)

    axes[0, 0].plot(h[1][1:], h[0], lw=2, alpha=0.8)
    if out_lim is not None:
        axes[0, 0].set(xlim=(out_lim[0], out_lim[1]))

    axes[0, 0].set(title="Output", yticklabels=[])
    axes[0, 0].axvline(np.mean(ys), ls="--", lw=2, label="computed mean")
    axes[0, 0].axvline(func(x0), lw=1, label="actual mean")

    axes[0, 1].set_visible(False)

    # plot transfer function
    x = np.arange(in_lims[0], in_lims[1], 0.1)
    y = func(x)
    isct = func(x0)

    axes[1, 0].plot(x, y, "k")
    axes[1, 0].plot([x0, x0, in_lims[1]], [out_lim[1], isct, isct], color="r", lw=1)
    axes[1, 0].set(xlim=in_lims, ylim=out_lim, title=f"f(x) = {func_name}")

    # plot input
    h = np.histogram(data, num_bins, density=True)

    axes[1, 1].plot(h[0], h[1][1:], lw=2)
    axes[1, 1].set(title="Input", xticklabels=[])


def plot_distributed_scatters(data, f, N, figsize=(6, 3)):
    _, axes = ax.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].scatter(data[:N], range(N), alpha=0.2, s=1)
    axes[0].set(title="Input")

    axes[1].scatter(f(data[:N]), range(N), alpha=0.2, s=1)
    axes[1].set(title="Output")


def plot_bivariate_colormap(ax, xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    values = np.vstack([xs, ys])
    kernel = stats.gaussian_kde(values)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    pos = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(kernel.evaluate(pos).T, X.shape)
    ax.imshow(np.rot90(Z), cmap=plt.cm.Greys, extent=[xmin, xmax, ymin, ymax])


def plot_monte_carlo_mean(
    axes, xs, ys, f, mean_fx, label, figsize=(8, 4), plot_colormap=True
):
    plot_bivariate_colormap(axes[0], xs, ys)

    axes[0].scatter(xs, ys, marker=".", alpha=0.02, color="k")
    axes[0].set(xlim=(-20, 20), ylim=(-20, 20))
    axes[0].grid(False)

    fxs, fys = f(xs, ys)
    computed_mean_x = np.average(fxs)
    computed_mean_y = np.average(fys)

    plot_bivariate_colormap(axes[1], fxs, fys)

    axes[1].scatter(fxs, fys, marker=".", alpha=0.02, color="k")
    axes[1].scatter(
        mean_fx[0], mean_fx[1], marker="v", s=figsize[1] * 30, c="r", label=label
    )
    axes[1].scatter(
        computed_mean_x,
        computed_mean_y,
        marker="*",
        s=figsize[1] * 20,
        c="b",
        label="Computed Mean",
    )

    axes[1].set(xlim=[-100, 100], ylim=[-10, 200])
    axes[1].grid(False)
    axes[1].legend()
    print(
        f"Difference in mean x={computed_mean_x - mean_fx[0]:.3f}, y={computed_mean_y - mean_fx[1]:.3f}"
    )

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from .plot_common import plot_zs
from .plot_kf import plot_kf


def show_markov_chain(figsize=(4, 4), facecolor="w"):
    """Show a markov chain showing relative probability of an object turning"""

    _ = plt.figure(figsize=figsize, facecolor=facecolor)
    ax = plt.axes((0, 0, 1, 1), xticks=[], yticks=[], frameon=False)

    box_bg = "#DDDDDD"

    kf1c = patches.Circle((4, 5), 0.5, fc=box_bg)
    kf2c = patches.Circle((6, 5), 0.5, fc=box_bg)
    ax.add_patch(kf1c)
    ax.add_patch(kf2c)

    ax.text(4, 5, "Straight", ha="center", va="center", fontsize="medium")
    ax.text(6, 5, "Turn", ha="center", va="center", fontsize="medium")

    # btm
    ax.text(5, 3.9, ".05", ha="center", va="center", fontsize="large")
    ax.annotate(
        "",
        xy=(4.1, 4.5),
        xycoords="data",
        xytext=(6, 4.5),
        textcoords="data",
        size=10,
        arrowprops=dict(arrowstyle="->", ec="k", connectionstyle="arc3,rad=-0.5"),
    )
    # top
    ax.text(5, 6.1, ".03", ha="center", va="center", fontsize="large")
    ax.annotate(
        "",
        xy=(6, 5.5),
        xycoords="data",
        xytext=(4.1, 5.5),
        textcoords="data",
        size=10,
        arrowprops=dict(arrowstyle="->", ec="k", connectionstyle="arc3,rad=-0.5"),
    )

    ax.text(3.5, 5.6, ".97", ha="center", va="center", fontsize="large")
    ax.annotate(
        "",
        xy=(3.9, 5.5),
        xycoords="data",
        xytext=(3.55, 5.2),
        textcoords="data",
        size=10,
        arrowprops=dict(
            arrowstyle="->", ec="k", connectionstyle="angle3,angleA=150,angleB=0"
        ),
    )

    ax.text(6.5, 5.6, ".95", ha="center", va="center", fontsize="large")
    ax.annotate(
        "",
        xy=(6.1, 5.5),
        xycoords="data",
        xytext=(6.45, 5.2),
        textcoords="data",
        size=10,
        arrowprops=dict(
            arrowstyle="->",
            fc="0.2",
            ec="k",
            connectionstyle="angle3,angleA=-150,angleB=2",
        ),
    )

    ax.axis("equal")


def plot_adkf_2d(
    axes, xs, z_xs2, dt, Q_scale_factor, std_scale, std_title=False, Q_title=False
):
    plot_zs(axes[0], z_xs2, dt=dt, label="z")
    plot_kf(axes[0], xs[:, 0], dt=dt, lw=1.5)
    axes[0].set(
        xlabel="time",
        ylabel="ϵ",
        title=f"position (std={std_scale}, Q scale={Q_scale_factor})",
    )

    axes[1].plot(np.arange(0, len(xs) * dt, dt), xs[:, 1], lw=1.5)
    axes[1].set(
        xlabel="time", title=f"velocity (std={std_scale}, Q scale={Q_scale_factor})"
    )

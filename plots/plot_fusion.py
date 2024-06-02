import numpy as np

from .plot_common import plot_zs
from .plot_kf import plot_kf


def plot_fusion_kf2d(ax, saver):
    ts = np.arange(0.1, 10, 0.1)
    plot_zs(ax, ts, saver.z[:, 0])
    ax.plot(ts, saver.z[:, 1], ls="--", label="Sensor")
    plot_kf(
        ax,
        ts,
        saver.x[:, 0],
    )
    ax.legend(loc=4)
    ax.set(xlabel="time", ylabel="meters", ylim=(0, 100))


def plot_fusion_kf(axes, xs, ts, zs_pos, zs_vel):
    ys = np.array(xs)

    axes[0].plot(
        zs_pos[:, 0],
        zs_pos[:, 1],
        ls="--",
        label="Pos Sensor",
    )
    plot_kf(axes[0], ts, ys[:, 0], label="Kalman filter")
    axes[0].set(title="Position", ylabel="meters")
    axes[0].legend()

    plot_zs(axes[1], zs_vel[:, 0], zs_vel[:, 1], label="Vel Sensor")
    plot_kf(axes[1], ts, ys[:, 1], label="Kalman filter")
    axes[1].set(title="Velocity", ylabel="meters", xlabel="time")
    axes[1].legend()

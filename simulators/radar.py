import math
import sys

import numpy as np
import sympy as sy
from numpy import linalg, random

sys.path.append("..")
from plots.plot_kf import plot_kf, plot_track


class RadarStation:
    def __init__(self, pos, range_std, elev_std):
        self.pos = np.asarray(pos)
        self.range_std = range_std
        self.elev_std = elev_std

    def reading_of(self, ac_pos):
        """Returns (range, elevation) to aircraft.
        Elevation is in radians.
        """
        diff = np.subtract(ac_pos, self.pos)
        rng = linalg.norm(diff)
        brg = math.atan2(diff[1], diff[0])
        return rng, brg

    def noisy_reading(self, ac_pos):
        """Compute range and elev angle to aircraft with simulated noise"""
        rng, brg = self.reading_of(ac_pos)
        rng += random.randn() * self.range_std
        brg += random.randn() * self.elev_std
        return rng, brg


class ACSim:
    def __init__(self, pos, vel, vel_std):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.vel_std = vel_std

    def update(self, dt):
        """Compute and returns next position. Incorporates random variation in velocity."""
        dx = self.vel * dt + (random.randn() * self.vel_std) * dt
        self.pos += dx
        return self.pos


class RadarSim:
    """Simulates the radar signal returns from an object
    flying at a constant altityude and velocity in 1D.
    """

    def __init__(self, dt, pos, vel, alt):
        self.pos = pos
        self.vel = vel
        self.alt = alt
        self.dt = dt

    def get_range(self):
        """Returns slant range to the object.
        Call once for each new measurement at dt time from last call.
        """
        # add some process noise to the system
        self.vel = self.vel + 0.1 * random.randn()
        self.alt = self.alt + 0.1 * random.randn()
        self.pos = self.pos + self.vel * self.dt
        # add measurement noise
        err = self.pos * 0.05 * random.randn()
        range_ = np.hypot(self.pos, self.alt)
        return range_ + err


def HJac3d(x):
    """Compute Jacobian of H matrix at x"""
    if type(x) == list:
        x = np.array(x)

    horiz_dist = x[0]
    altitude = x[2]
    denom = np.hypot(horiz_dist, altitude)
    return np.array([[horiz_dist / denom, 0.0, altitude / denom]])


def H2dRE(x, refpos=(0, 0)):
    if type(x) == list | tuple:
        x = np.array(x)

    dx = x[0] - refpos[0]
    dy = x[1] - refpos[1]
    range_ = np.hypot(dx, dy)
    elev = math.atan2(dy, dx)
    return range_, elev


def H3dRE(x, refpos=(0, 0)):
    if type(x) == list | tuple:
        x = np.array(x)

    dx = x[0] - refpos[0]
    dz = x[2] - refpos[1]
    range_ = np.hypot(dx, dz)
    elev = math.atan2(dz, dx)
    return range_, elev


def H4dRE(x, refpos=(0, 0)):
    if type(x) == list | tuple:
        x = np.array(x)

    dx = x[0] - refpos[0]
    dz = x[2] - refpos[1]
    range_ = np.hypot(dx, dz)
    elev = math.atan2(dz, dx)
    return range_, elev, x[1], x[3]


def hx(x):
    """compute measurement for slant range that would correspond to state x."""
    return np.hypot(x[0], x[2])


def plot_radar3d(axes, time, xs, ylabels=None, track=None):
    if ylabels is None:
        ylabels = ["position", "velocity", "altitude"]

    ys_dict = dict(zip(ylabels, [xs[:, 0], xs[:, 1], xs[:, 2]]))

    if track is not None:
        track_dict = dict(zip(ylabels, [track[:, 0], track[:, 1], track[:, 2]]))

    for ax, ylabel in zip(axes, ylabels):
        plot_kf(ax, time, ys=ys_dict[ylabel], label=f"{ylabel} filtered")
        if track is not None:
            plot_track(ax, time, ys=track_dict[ylabel], label=f"{ylabel} track")
        ax.set(xlabel="time", ylabel=ylabel)
        ax.legend()
        ax.grid(True)


def plot_radar3d_zs(ax, time, xs, obj, track=None, ylabels=None):
    xs = np.asarray(xs)

    if ylabels is None:
        ylabels = ["position", "velocity", "altitude"]
    ys_dict = dict(
        zip(["position", "velocity", "altitude"], [xs[:, 0], xs[:, 1], xs[:, 2]])
    )

    ax.plot(time, ys_dict[obj], label=f"filtered {obj}")
    if track:
        ax.plot(time, track, label="track", lw=2, ls="--", c="k")
    ax.set(xlabel="time", ylabel=f"{obj}")
    ax.legend()


def plot_moving_target(ax):
    pos = np.array([5.0, 5.0])
    for _ in range(5):
        pos += (0.5, 1.0)
        actual_angle = math.atan2(pos[1], pos[0])
        d = np.sqrt(pos[0] ** 2 + pos[1] ** 2)

        xs, ys = [], []
        for _ in range(100):
            a = actual_angle + random.randn() * math.radians(1)
            xs.append(d * math.cos(a))
            ys.append(d * math.sin(a))

        ax.scatter(xs, ys, c="C0")

    ax.plot([5.5, pos[0]], [6, pos[1]], c="g", linestyle="--")


def plot_iscts_two_sensors(axes):
    poss = [np.array([4.0, 4]), np.array([3.0, 3.0])]
    sas = [[0, 2], [3, 4]]
    sbs = [[8, 2], [3, 7]]
    Ns = [4, 5]

    for ax, pos, sa, sb, N in zip(axes, poss, sas, sbs, Ns):
        ax.scatter(*sa, s=200, marker="v", c="k", label="Sensor 1")
        ax.scatter(*sb, s=200, marker="s", c="C0", label="Sensor 2")
        _plot_iscts(ax, pos, sa, sb, N=N)
        ax.legend()


def _plot_iscts(ax, pos, sa, sb, N=4):
    for _ in range(N):
        pos += (0.5, 1.0)
        actual_angle_a = math.atan2(pos[1] - sa[1], pos[0] - sa[0])
        actual_angle_b = math.atan2(pos[1] - sb[1], pos[0] - sb[0])

        da = np.sqrt((sa[0] - pos[0]) ** 2 + (sa[1] - pos[1]) ** 2)
        db = np.sqrt((sb[0] - pos[0]) ** 2 + (sb[1] - pos[1]) ** 2)

        xs, ys, xs_a, xs_b, ys_a, ys_b = [], [], [], [], [], []

        for _ in range(300):
            aa = actual_angle_a + random.randn() * math.radians(1)
            ab = actual_angle_b + random.randn() * math.radians(1)

            xs_a.append(da * math.cos(aa) + sa[0])
            ys_a.append(da * math.sin(aa) + sa[1])

            xs_b.append(db * math.cos(ab) + sb[0])
            ys_b.append(db * math.sin(ab) + sb[1])

            x, y = _isct(sa, sb, aa, ab)
            xs.append(x)
            ys.append(y)

        ax.scatter(xs_a, ys_a, c="k", edgecolor="k")
        ax.scatter(xs_b, ys_b, marker="v", edgecolor=None, c="C0")
        ax.scatter(xs, ys, marker=".", c="r", alpha=0.5)


def _isct(sa, sb, aa, ab):
    """Returns the (x, y) intersections of points pa and pb given the bearing aa for point pa and bearing ab for point pb."""
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    # bearing to angle
    # aa = 90 - aa
    # ab = 90 - ab

    # Line AB represented as a1x + b1y = c1
    # Line CD represented as a2x + b2y = c2
    a1, a2 = math.sin(aa), math.sin(ab)
    b1, b2 = -math.cos(aa), -math.cos(ab)
    c1, c2 = a1 * sa[0] + b1 * sa[1], a2 * sb[0] + b2 * sb[1]

    det = a1 * b2 - a2 * b1

    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det

    return x, y


def show_radar_chart(ax):
    ax.scatter([1, 2], [1, 2])
    ax.annotate(
        "",
        xy=(2, 2),
        xytext=(1, 1),
        arrowprops=dict(arrowstyle="->", ec="r", shrinkA=3, shrinkB=4),
    )
    ax.annotate(
        "",
        xy=(2, 1),
        xytext=(1, 1),
        arrowprops=dict(arrowstyle="->", ec="b", shrinkA=0, shrinkB=0),
    )
    ax.annotate(
        "",
        xy=(2, 2),
        xytext=(2, 1),
        arrowprops=dict(arrowstyle="->", ec="b", shrinkA=0, shrinkB=4),
    )

    ax.annotate("ϵ", xy=(1.2, 1.05), color="b")
    ax.annotate("Aircraft", xy=(2.04, 2.0), color="b")
    ax.annotate("altitude (y)", xy=(2.04, 1.5), color="k")
    ax.annotate("x", xy=(1.5, 0.9))
    ax.annotate("Radar", xy=(0.95, 0.8))
    ax.annotate("Slant\n  (r)", xy=(1.5, 1.62), color="r")

    ax.set(xlim=[0.9, 2.5], ylim=[0.5, 2.5])
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.set(frame_on=False)


def radar3d_symbol():
    x, x_vel, y = sy.symbols("x, x_vel y")
    H = sy.Matrix([sy.sqrt(x**2 + y**2)])
    state = sy.Matrix([x, x_vel, y])
    J = H.jacobian(state)
    print(state)
    print(J)

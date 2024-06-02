import math
import sys

import numpy as np
import sympy as sy
from numpy import linalg, random

from .datagen import gen_particles_gaussian, gen_particles_uniform

sys.path.append("..")
from filters import kalman_ekf, particle, resamplers
from plots import plot_pf


def robot_move(x, dt, u, wheelbase):
    hdg = x[2]
    vel = u[0]
    steering_angle = u[1]
    dist = vel * dt

    if abs(steering_angle) <= 0.001:
        return x + np.array([dist * math.cos(hdg), dist * math.sin(hdg), 0])

    β = (dist / wheelbase) * math.tan(steering_angle)
    r = wheelbase / math.tan(steering_angle)
    sinh, sinhb = math.sin(hdg), math.sin(hdg + β)
    cosh, coshb = math.cos(hdg), math.cos(hdg + β)
    return x + np.array([-r * sinh + r * sinhb, r * cosh - r * coshb, β])


def robot_turn(v, t0, t1, steps):
    return [[v, a] for a in np.linspace(np.radians(t0), np.radians(t1), steps)]


def normalize_angle(x):
    """Normalize angle to be in range [-pi, pi)"""
    x = x % (2 * np.pi)
    if x > np.pi:
        x -= 2 * np.pi
    return x


def residual(a, b):
    """Compute residual (a-b) between measurements containing
    [range, bearing]. Bearing is normalized to [-pi, pi)"""
    y = a - b
    y[1] = normalize_angle(y[1])
    return y


def residual_h(a, b):
    y = a - b
    for i in range(0, len(y), 2):
        y[i + 1] = normalize_angle(y[i + 1])
    return y


def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    return y


def Hx(x, landmarks):
    """Takes a state variable and returns the measurement that would correspond to that state."""
    hx = []
    for lmark in landmarks:
        dx, dy = lmark[0] - x[0], lmark[1] - x[1]
        dist = np.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        hx += [dist, normalize_angle(angle - x[2])]
    return np.array(hx)


def Hx_1(x, lmark):
    """Takes a state variable and returns the measurement that would correspond to that state."""
    dx, dy = lmark[0] - x[0], lmark[1] - x[1]
    dist = np.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    hx = [dist, normalize_angle(angle - x[2])]
    return np.array(hx)


def HJac(x, lmark):
    """Compute Jacobian of H matrix where h(x) computes the range and bearing to a landmark for state x"""

    dx, dy = lmark[0] - x[0, 0], lmark[1] - x[1, 0]
    dist = np.hypot(dx, dy)

    return np.array(
        [
            [-dx / dist, -dy / dist, 0],
            [dy / dist**2, -dx / dist**2, -1],
        ]
    )


def z_landmark(lmark, sim_pos, std_rng, std_brg):
    x, y = sim_pos[0], sim_pos[1]
    dist = np.hypot(lmark[0] - x, lmark[1] - y)
    angle = math.atan2(lmark[1] - y, lmark[0] - x) - sim_pos[2]
    return np.array(
        [dist + random.randn() * std_rng, angle + random.randn() * std_brg],
    )


def state_mean(sigmas, Wm):
    x = np.zeros(3)
    sum_sin = np.sum(np.sin(sigmas[:, 2]) @ Wm)
    sum_cos = np.sum(np.cos(sigmas[:, 2]) @ Wm)
    x[0] = np.sum(sigmas[:, 0] @ Wm)
    x[1] = np.sum(sigmas[:, 1] @ Wm)
    x[2] = math.atan2(sum_sin, sum_cos)
    return x


def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]
    x = np.zeros(z_count)
    for z in range(0, z_count, 2):
        sum_sin = np.sum(np.sin(sigmas[:, z + 1]) @ Wm)
        sum_cos = np.sum(np.cos(sigmas[:, z + 1]) @ Wm)
        x[z] = np.sum(sigmas[:, z] @ Wm)
        x[z + 1] = math.atan2(sum_sin, sum_cos)
    return x


def robot3d_symbol(print=False):
    vars = sy.symbols("a, x, y, v, w, θ, t")
    [a, x, y, v, w, θ, time] = vars
    d = v * time
    β = (d / w) * sy.tan(a)
    r = w / sy.tan(a)

    fxu = sy.Matrix(
        [
            [x - r * sy.sin(θ) + r * sy.sin(θ + β)],
            [y + r * sy.cos(θ) - r * sy.cos(θ + β)],
            [θ + β],
        ]
    )
    F_j = fxu.jacobian(sy.Matrix([x, y, θ]))
    V_j = fxu.jacobian(sy.Matrix([v, a]))

    if print:
        print("fxu: ", fxu)
        print("F_j: ", fxu)
        print("V_j: ", fxu)

    return fxu, F_j, V_j, vars


class RobotEKF(kalman_ekf.ExtendedKalmanFilter):
    def __init__(self, dt, wheelbase, std_vel, std_steer):
        kalman_ekf.ExtendedKalmanFilter.__init__(self, 3, 2, 2)
        self.dt = dt
        self.wheelbase = wheelbase
        self.std_vel = std_vel
        self.std_steer = std_steer

        fxu, F_j, V_j, vars = robot3d_symbol()
        self.fxu = fxu
        self.F_j = F_j
        self.V_j = V_j
        [a, x, y, v, w, θ, time] = vars

        # save dictionary and it's variables for later use
        self.subs = {x: 0, y: 0, v: 0, a: 0, time: dt, w: wheelbase, θ: 0}
        self.x_x, self.x_y = x, y
        self.v, self.a, self.θ = v, a, θ

    def predict(self, u):
        self.x = self.move(self.x, u, self.dt)
        self.subs[self.θ] = self.x[2, 0]
        self.subs[self.v] = u[0]
        self.subs[self.a] = u[1]

        F = np.array(self.F_j.evalf(subs=self.subs)).astype(float)
        V = np.array(self.V_j.evalf(subs=self.subs)).astype(float)

        # covariance of motion noise in control space
        M = np.array(
            [
                [self.std_vel * u[0] ** 2, 0],
                [0, self.std_steer**2],
            ]
        )
        self.P = F @ self.P @ F.T + V @ M @ V.T

    def move(self, x, u, dt):
        hdg = x[2, 0]
        vel = u[0]
        steering_angle = u[1]
        dist = vel * dt

        if abs(steering_angle) > 0.001:
            β = (dist / self.wheelbase) * math.tan(steering_angle)
            r = self.wheelbase / math.tan(steering_angle)
            dx = np.array(
                [
                    [-r * math.sin(hdg) + r * math.sin(hdg + β)],
                    [r * math.cos(hdg) - r * math.cos(hdg + β)],
                    [β],
                ]
            )
        else:
            dx = np.array(
                [
                    [dist * math.cos(hdg)],
                    [dist * math.sin(hdg)],
                    [0],
                ]
            )
        return x + dx


def robot_pf(
    ax,
    N,
    iters=18,
    sensor_std_err=0.1,
    initial_x=None,
    show_particles=False,
):
    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    NL = len(landmarks)

    # create particles and weights
    if initial_x is not None:
        particles = gen_particles_gaussian(mean=initial_x, std=(5, 5, np.pi / 4), N=N)
    else:
        particles = gen_particles_uniform((0, 20), (0, 20), (0, 6.28), N)
    weights = np.ones(N) / N

    if show_particles:
        plot_pf.plot_particles(ax, particles, markersize=25)

    xs = []
    robot_pos = np.array([0.0, 0.0])
    for _ in range(iters):
        robot_pos += (1, 1)
        # distance from robot to each landmark
        zs = linalg.norm(landmarks - robot_pos, axis=1) + (
            random.randn(NL) * sensor_std_err
        )
        # move diagonally forward to (x+1, x+1)
        particle.pf_predict(particles, u=(0.00, 1.414), std=(0.2, 0.05))
        # incorporate measurements
        particle.pf_update(
            particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks
        )
        # resample if too few effective particles
        if particle.neff(weights) < N / 2:
            indexes = resamplers.systematic_resample(weights)
            resamplers.resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1 / N)
        mu, var = particle.pf_estimate(particles, weights)
        xs.append(mu)

        if show_particles:
            plot_pf.plot_particles(ax, particles, color="k", marker=",")

        p1 = ax.scatter(robot_pos[0], robot_pos[1], marker="+", color="k", s=180, lw=3)
        p2 = ax.scatter(mu[0], mu[1], marker="s", color="r")

    ax.legend([p1, p2], ["Actual", "PF"], loc=4, numpoints=1)
    xs = np.array(xs)
    print("final position error, variance:\n\t", mu - np.array([iters, iters]), var)

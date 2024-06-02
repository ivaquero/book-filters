import math
import sys

import numpy as np
from numpy import random

sys.path.append("..")
from models import constant_velocity


def angle_between(x, y):
    return min(y - x, y - x + 360, y - x - 360, key=abs)


class ManeuveringTarget:
    def __init__(self, x0, y0, v0, heading):
        self.x = x0
        self.y = y0
        self.vel = v0
        self.hdg = heading
        self.cmd_vel = v0
        self.cmd_hdg = heading
        self.vel_step = 0
        self.hdg_step = 0
        self.vel_delta = 0
        self.hdg_delta = 0

    def update(self):
        vx = self.vel * math.cos(math.radians(90 - self.hdg))
        vy = self.vel * math.sin(math.radians(90 - self.hdg))
        self.x += vx
        self.y += vy

        if self.hdg_step > 0:
            self.hdg_step -= 1
            self.hdg += self.hdg_delta

        if self.vel_step > 0:
            self.vel_step -= 1
            self.vel += self.vel_delta
        return (self.x, self.y)

    def set_commanded_heading(self, hdg_degrees, steps):
        self.cmd_hdg = hdg_degrees
        self.hdg_delta = angle_between(self.cmd_hdg, self.hdg) / steps
        self.hdg_step = steps if abs(self.hdg_delta) > 0 else 0

    def set_commanded_speed(self, speed, steps):
        self.cmd_vel = speed
        self.vel_delta = (self.cmd_vel - self.vel) / steps
        self.vel_step = steps if abs(self.vel_delta) > 0 else 0


class NoisySensor:
    def __init__(self, std_noise=1.0):
        self.std = std_noise

    def sense(self, pos):
        """Pass in actual position as tuple (x, y).
        Returns position with noise added (x,y)"""
        return (
            pos[0] + (random.randn() * self.std),
            pos[1] + (random.randn() * self.std),
        )


def simulate_maneuver(steady_count, std):
    t = ManeuveringTarget(x0=0, y0=0, v0=0.3, heading=0)
    xs, ys = [], []

    for _ in range(30):
        x, y = t.update()
        xs.append(x)
        ys.append(y)

    t.set_commanded_heading(310, 25)
    t.set_commanded_speed(1, 15)

    for _ in range(steady_count):
        x, y = t.update()
        xs.append(x)
        ys.append(y)

    ns = NoisySensor(std)
    pos = np.array(list(zip(xs, ys)))
    zs = np.array([ns.sense(p) for p in pos])
    return pos, zs


def simulate_turning(N=600, turn_start=400):
    """simulate a moving target"""

    # r = 1.
    dt = 1.0
    phi_sim = constant_velocity.FCV(4, dt)

    γ = np.array(
        [[dt**2 / 2, 0], [dt, 0], [0, dt**2 / 2], [0, dt]],
    )

    x = np.array([[2000, 0, 10000, -15.0]]).T

    simxs = []
    for i in range(N):
        x = phi_sim @ x
        if i >= turn_start:
            x += γ @ np.array([[0.075, 0.075]]).T
        simxs.append(x)

    return np.array(simxs)

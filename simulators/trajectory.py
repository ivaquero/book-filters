import math
import sys

import numpy as np
from numpy import random

sys.path.append("..")
from filters.solvers import rk4


def xvel(x, t, acc=0):
    return xvel.vel + acc * t


def yvel(x, t, acc=-9.8):
    return yvel.vel + acc * t


class BallTrajectory2D:
    def __init__(self, x0, y0, vel, theta_deg=0.0, g=9.8, noise=None):
        if noise is None:
            noise = [0.0, 0.0]
        self.x = x0
        self.y = y0
        self.t = 0
        theta = math.radians(theta_deg)
        xvel.vel = math.cos(theta) * vel
        yvel.vel = math.sin(theta) * vel
        self.g = g
        self.noise = noise

    def step(self, dt):
        self.x = rk4(self.x, self.t, dt, xvel)
        self.y = rk4(self.y, self.t, dt, yvel)
        self.t += dt
        return (
            self.x + random.randn() * self.noise[0],
            self.y + random.randn() * self.noise[1],
        )


class BaseballPath:
    def __init__(self, x0, y0, launch_angle_deg, vel_ms, noise=(1.0, 1.0)):
        """Create 2D baseball path object"""
        omega = math.radians(launch_angle_deg)
        self.v_x = vel_ms * math.cos(omega)
        self.v_y = vel_ms * math.sin(omega)
        self.x = x0
        self.y = y0
        self.noise = noise

    def drag_force(self, vel):
        """Returns the force on a baseball due to air drag at
        the specified vel. Units are SI
        """
        B_m = 0.0039 + 0.0058 / (1.0 + math.exp((vel - 35.0) / 5.0))
        return B_m * vel

    def update(self, dt, vel_wind=0.0):
        """Compute the ball position based on the specified time
        step and wind vel. Returns (x, y) position tuple.
        """
        # Euler equations for x and y
        self.x += self.v_x * dt
        self.y += self.v_y * dt
        # force due to air drag
        v_x_wind = self.v_x - vel_wind
        v = np.sqrt(v_x_wind**2 + self.v_y**2)
        F = self.drag_force(v)
        # Euler's equations for vel
        self.v_x = self.v_x - F * v_x_wind * dt
        self.v_y = self.v_y - 9.81 * dt - F * self.v_y * dt
        return (
            self.x + random.randn() * self.noise[0],
            self.y + random.randn() * self.noise[1],
        )

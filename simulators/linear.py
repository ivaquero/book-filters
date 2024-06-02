import numpy as np
from numpy import random


class DogSimulation:
    def __init__(self, x0=0, model_mean=1, Q=0, R=0):
        self.x = x0
        self.model_mean = model_mean
        self.p_std = np.sqrt(Q)
        self.z_std = np.sqrt(R)

    def move(self, dt=1.0, acc=0):
        model_mean = self.model_mean + random.randn() * self.p_std
        self.x += model_mean * dt
        self.model_mean += acc
        return self.x

    def sense_position(self):
        return self.x + random.randn() * self.z_std

    def move_and_sense(self, dt=1.0, acc=0):
        self.move(dt, acc)
        return self.sense_position()

    def simulate(self, dt=1, num=1, acc=0):
        xs = np.array([self.move(dt, acc) for _ in range(num)])
        zs = np.array([self.move_and_sense(dt) for _ in range(num)])
        return xs, zs


class PosSensor:
    def __init__(self, pos=(0, 0), vel=(0, 0), R=1.0):
        self.vel = vel
        self.pos = [pos[0], pos[1]]
        self.R = R

    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]

        return [
            self.pos[0] + random.randn() * self.R,
            self.pos[1] + random.randn() * self.R,
        ]


class CVObject:
    def __init__(self, x0=0, vel=1.0, Q=0.06, R=0.1):
        self.x = x0
        self.vel = vel
        self.Q = Q
        self.R = R

    def update(self):
        self.vel += random.randn() * self.Q
        self.x += self.vel
        return (self.x, self.vel)

    def sense(self):
        return self.x + random.randn() * self.R

    def simulate(self, count):
        xs, zs = [], []
        for _ in range(count):
            x = self.update()
            z = self.sense()
            xs.append(x)
            zs.append(z)
        return np.array(xs), np.array(zs)


class CAObject:
    def __init__(self, x0=0, vel=1.0, acc=0.1, Q=0.1, R=0.1):
        self.x = x0
        self.vel = vel
        self.acc = acc
        self.Q = Q
        self.R = R

    def update(self):
        self.acc += random.randn() * self.Q
        self.vel += self.acc
        self.x += self.vel
        return (self.x, self.vel, self.acc)

    def sense(self):
        return self.x + random.randn() * self.R

    def simulate(self, count):
        xs, zs = [], []
        for _ in range(count):
            x = self.update()
            z = self.sense()
            xs.append(x)
            zs.append(z)
        return np.asarray(xs), zs

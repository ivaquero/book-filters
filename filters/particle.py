import numpy as np
from numpy import random
from scipy import stats


class ParticleFilter:
    def __init__(self, N, x_dim, y_dim):
        self.particles = np.empty((N, 3))  # x, y, heading
        self.N = N
        self.x_dim = x_dim
        self.y_dim = y_dim

        # distribute particles randomly with uniform weight
        self.weights = np.empty(N)
        self.weights.fill(1.0 / N)
        self.particles[:, 0] = random.uniform(0, x_dim, size=N)
        self.particles[:, 1] = random.uniform(0, y_dim, size=N)
        self.particles[:, 2] = random.uniform(0, 2 * np.pi, size=N)

    def predict(self, u, std, dt=1):
        """Move according to control input u with noise std"""
        self.particles[:, 2] += u[0] + random.randn(self.N) * std[0]
        self.particles[:, 2] %= 2 * np.pi

        dist = (u[1] * dt) + (random.randn(self.N) * std[1])
        self.particles[:, 0] += np.cos(self.particles[:, 2]) * dist
        self.particles[:, 1] += np.sin(self.particles[:, 2]) * dist

        self.particles[:, 0:2] += u + random.randn(self.N, 2) * std

    def weight(self, z, var):
        dist = np.sqrt(
            (self.particles[:, 0] - z[0]) ** 2 + (self.particles[:, 1] - z[1]) ** 2
        )

        # simplification assumes variance is invariant to world projection
        n = stats.norm(0, np.sqrt(var))
        prob = n.pdf(dist)

        # particles far from a measurement will give us 0.0 for a probability due to floating point limits. Once we hit zero we can never recover, so add some small nonzero value to all points.
        prob += 1.0e-12
        self.weights += prob
        self.weights /= sum(self.weights)  # normalize

    def neff(self):
        return 1.0 / np.sum(np.square(self.weights))

    def resample(self):
        p = np.zeros((self.N, 3))
        w = np.zeros(self.N)

        cumsum = np.cumsum(self.weights)
        for i in range(self.N):
            index = np.searchsorted(cumsum, random.random())
            p[i] = self.particles[index]
            w[i] = self.weights[index]

        self.particles = p
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        """Returns mean and variance"""
        pos = self.particles[:, 0:2]
        mu = np.average(pos, weights=self.weights, axis=0)
        var = np.average((pos - mu) ** 2, weights=self.weights, axis=0)

        return mu, var


def neff(weights):
    return 1.0 / np.sum(np.square(weights))


def pf_estimate(particles, weights):
    """Returns mean and variance of the weighted particles"""
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def pf_update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= stats.norm(distance, R).pdf(z[i])
        weights += 1.0e-300  # avoid round-off to zero
        weights /= sum(weights)  # normalize


def pf_predict(particles, u, std, dt=1.0):
    """Move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)"""
    N = len(particles)

    # update heading
    particles[:, 2] += u[0] + (random.randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (random.randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

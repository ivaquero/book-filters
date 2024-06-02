import numpy as np
from numpy import random


def gen_cvca(
    num: int,
    x0: float,
    dx: float,
    ddx: float = 0,
    dt: float = 1,
    R: float = 1,
    random_func: callable = random.randn,
):
    """Generate samples from a constant velocity and constant acceleration model.

    Args:
        num (int): number of samples
        x0 (float): initial state
        dx (float): velocity
        ddx (float, optional): acceleration. Defaults to 0.
        dt (float, optional): time step. Defaults to 1.
        R (float, optional): random noise variance. Defaults to 1.
        random_func (callable, optional): random noise function. Defaults to random.randn.

    Returns:
        tuple: predictions and measurements
    """
    x = x0
    xs, zs = [], []
    for i in range(num):
        x += dx * dt
        xs.append(x)
        x += ddx * (i**2) / 2 + dx * i
        zs.append(x + random_func() * np.sqrt(R))
        dx += ddx
    return np.array(xs), np.asarray(zs)


def gen_jittered_vel(
    num: int,
    x0: float,
    dx: float,
    dt: float = 1,
    Q: float = 0,
    R: float = 1,
    random_func: callable = random.randn,
):
    """_summary_

    Args:
        num (int): number of samples
        x0 (float): initial state
        dx (float): base velocity
        dt (float, optional): time step. Defaults to 1.
        Q (float): system noise variance. Defaults to 0.
        R (float): random noise variance. Defaults to 1.
        random_func (callable, optional): random noise function. Defaults to random.randn.

    Returns:
        tuple: predictions and measurements
    """
    x = x0
    xs, zs = [], []
    for _ in range(num):
        x += (dx + (random_func() * np.sqrt(Q))) * dt
        xs.append(x)
        zs.append(x + random_func() * np.sqrt(R))
    return np.array(xs), np.asarray(zs)


def jitterfy(center: np.array, std: float) -> np.array:
    """_summary_

    Args:
        center (np.array): origen array
        std (float): jitterfy factor

    Returns:
        np.array: jittered array
    """
    return center + (random.randn() * std)


def gen_sensor_data(time, pos_std, vel_std, seed=1123):
    random.seed(seed)
    pos_data, vel_data = [], []
    dt = 0.0
    for _ in range(time * 3):
        dt += 1 / 3.0
        time_jittered = dt + random.randn() * 0.01
        pos_data.append(
            [time_jittered, time_jittered + random.randn() * pos_std],
        )

    dt = 0.0
    for _ in range(time * 7):
        dt += 1 / 7.0
        time_jittered = dt + random.randn() * 0.006
        vel_data.append([time_jittered, 1.0 + random.randn() * vel_std])
    return pos_data, vel_data


def gen_particles_uniform(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = random.uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


def gen_particles_gaussian(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (random.randn(N) * std[0])
    particles[:, 1] = mean[1] + (random.randn(N) * std[1])
    particles[:, 2] = mean[2] + (random.randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

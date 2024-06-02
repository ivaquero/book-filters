import numpy as np
from numpy import random


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))


def residual_resample(weights):
    """Performs the residual resampling algorithm used by particle filters.

    Based on observation that we don't need to use random numbers to select most of the weights. Take int(N*w^i) samples of each particle i, and then resample any remaining using a standard resampling algorithm.

    References
    ----------
    J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic systems. Journal of the American Statistical Association, 93(443):1032–1044, 1998.
    """

    N = len(weights)
    indexes = np.zeros(N, "i")

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N * np.asarray(weights))).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):  # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    residual = weights - num_copies  # get fractional part
    residual /= sum(residual)  # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1.0  # avoid round-off errors: ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, random.random(N - k))

    return indexes


def stratified_resample(weights):
    """Performs the stratified resampling algorithm used by particle filters.

    This algorithms aims to make selections relatively uniformly across the particles. It divides the cumulative sum of the weights into N equal divisions, and then selects one particle randomly from each division. This guarantees that each sample is between 0 and 2/N apart.
    """

    N = len(weights)
    # make N subdivisions, and chose a random position within each one
    positions = (random.random(N) + range(N)) / N

    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def systematic_resample(weights):
    """Performs the systemic resampling algorithm used by particle filters.

    This algorithm separates the sample space into N divisions. A single random offset is used to to choose where to sample from for all divisions. This guarantees that every sample is exactly 1/N apart.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random.random() + np.arange(N)) / N

    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def multinomial_resample(weights):
    """This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the resampled point based on a uniformly distributed random number. Run time is O(n log n). You do not want to use this algorithm in practice; for some reason it is popular in blogs and online courses so I included it for reference.
    """
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # avoid round-off errors: ensures sum is exactly one
    return np.searchsorted(cumulative_sum, random.random(len(weights)))

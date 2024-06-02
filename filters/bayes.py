import itertools

import numpy as np

from .stats import gaussian, gaussian_product, gaussian_sum


def prior_with_jitter(belief, move, p_correct, p_under, p_over):
    n = len(belief)
    prior = np.zeros(n)
    for i in range(n):
        prior[i] = (
            belief[(i - move) % n] * p_correct
            + belief[(i - move - 1) % n] * p_over
            + belief[(i - move + 1) % n] * p_under
        )
    return prior


def prior_by_convol(pdf, offset, kernel):
    N = len(pdf)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i, k in itertools.product(range(N), range(kN)):
        index = (i + (width - k) - offset) % N
        prior[i] += pdf[index] * kernel[k]
    return prior


def likelihood(data, z, z_prob):
    try:
        odd = z_prob / (1.0 - z_prob)
    except ZeroDivisionError:
        odd = 1e8

    likelihood = np.ones(len(data))
    likelihood[data == z] *= odd
    return likelihood


def posterior(likelihood, prior):
    return likelihood * prior / sum(likelihood * prior)


def prior_estimate(x0, model, zs, R, show_steps=False):
    x = x0
    priors, estimates, ps = (
        np.zeros((len(zs), 2)),
        np.zeros((len(zs), 2)),
        np.zeros((len(zs), 2)),
    )
    for i, z in enumerate(zs):
        prior = gaussian_sum(x, model)
        likelihood = gaussian(z, R)
        x = gaussian_product(prior, likelihood)
        priors[i], estimates[i], ps[i] = prior, x.mean, x.var
        if show_steps:
            if i == 0:
                print_header
            print_steps(prior, z, x)
            if i == len(zs) - 1:
                print_conclusion(z, x)
    return (priors, estimates, ps)


def print_steps(prior, z, x):
    print(f"{prior[0]: 7.3f} {prior[1]: 8.3f}", end="\t")
    print(f"{z:.3f}\t{x[0]: 7.3f} {x[1]: 7.3f}")


def print_header():
    print("\tprior\t\t\tESTIMATE")
    print("    x\t    var\t\t  z\t    x\t   var")


def print_conclusion(z, x):
    print(f" final estimate:     {x.mean:10.3f}")
    print(f" actual final value: {z:10.3f}")

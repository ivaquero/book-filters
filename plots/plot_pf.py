import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import random

sys.path.append("..")
from filters.resamplers import residual_resample


def plot_particles(
    ax,
    particles,
    marker="o",
    markersize=1,
    color="g",
):
    N = len(particles)
    alpha = 0.20
    if N > 5000:
        alpha *= np.sqrt(5000) / np.sqrt(N)
    ax.scatter(
        particles[:, 0],
        particles[:, 1],
        alpha=alpha,
        marker=marker,
        s=markersize,
        color=color,
    )


def plot_pf(axes, pf, xlim=100, ylim=100, weights=True):
    if weights:
        axes[1].set(yticklabels="", xlim=(0, ylim), ylim=(0, xlim))
        axes[1].scatter(pf.particles[:, 0], pf.weights, marker=".", s=1, color="k")

        axes[2].set(xticklabels="", xlim=(0, ylim), ylim=(0, xlim))
        axes[2].scatter(pf.weights, pf.particles[:, 1], marker=".", s=1, color="k")

    axes[0].scatter(pf.particles[:, 0], pf.particles[:, 1], marker=".", s=1, color="k")
    axes[0].set(xlim=(0, ylim), ylim=(0, xlim))


def plot_hbar(a, N, figsize):
    cmap = mpl.colors.ListedColormap(
        (
            [
                [0.0, 0.4, 1.0],
                [0.0, 0.8, 1.0],
                [1.0, 0.8, 0.0],
                [1.0, 0.4, 0.0],
            ]
            * (N // 4 + 1)
        )
    )
    cumsum = np.cumsum(np.asarray(a) / np.sum(a))
    cumsum = np.insert(cumsum, 0, 0)
    norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)

    _ = plt.figure(figsize=figsize)
    ax = plt.gcf().add_axes([0.01, 0.01, 1, 0.2])
    bar = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        drawedges=False,
        spacing="proportional",
        orientation="horizontal",
    )
    bar.set_ticks([])
    return ax, cumsum


def show_resample_multinomial(a, figsize=(6, 2)):
    N = len(a)
    ax, _ = plot_hbar(a, N, figsize=figsize)

    # make N subdivisions, and chose a random position within each one
    b = random.random(N)
    ax.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    ax.set(title="multinomial resampling")


def show_resample_stratified(a, figsize=(6, 2)):
    N = len(a)
    ax, _ = plot_hbar(a, N, figsize=figsize)

    xs = np.linspace(0.0, 1.0 - 1.0 / N, N)
    ax.vlines(xs, 0, 1, lw=2)

    # make N subdivisions, and chose a random position within each one
    b = (random.random(N) + range(N)) / N
    ax.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    ax.set(title="stratified resampling")


def show_resample_systematic(a, figsize=(6, 2)):
    N = len(a)
    ax, _ = plot_hbar(a, N, figsize=figsize)

    xs = np.linspace(0.0, 1.0 - 1.0 / N, N)
    ax.vlines(xs, 0, 1, lw=2)

    # make N subdivisions, and chose a random position within each one
    b = (random.random() + np.array(range(N))) / N
    ax.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    ax.set(title="systematic resampling")


def show_resample_residual(a, figsize=(6, 2)):
    N = len(a)
    ax, cumsum = plot_hbar(a, N, figsize=figsize)

    indexes = residual_resample(np.asarray(a) / np.sum(a))
    bins = np.bincount(indexes)
    for i in range(1, N):
        n = bins[i - 1]  # number particles in this sample
        if n > 0:
            b = np.linspace(cumsum[i - 1], cumsum[i], n + 2)[1:-1]
            plt.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    ax.set(title="residual resampling")

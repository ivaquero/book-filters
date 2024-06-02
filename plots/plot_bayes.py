import sys

import matplotlib.pyplot as plt
from scipy import stats

sys.path.append("..")
from filters.stats import gaussian_product


def plot_gaussian_product(ax, xs, g1, g2):
    g_product = gaussian_product(g1, g2)

    for g, ls in zip([g1, g2, g_product], ["-", "-", "--"]):
        ys = [stats.norm(g.mean, g.var**0.5).pdf(x) for x in xs]
        ax.plot(
            xs,
            ys,
            linestyle=ls,
            label=f"$N({g.mean:.2f}, {g.var:.2f})$",
        )

    ax.grid(1)
    ax.legend()


def plot_belief_prior(belief, prior, ylim, figwidth=8):
    _, axes = plt.subplots(
        1,
        2,
        figsize=(figwidth, figwidth / 2.5),
        constrained_layout=True,
        sharey=True,
    )
    for data, title, ax in zip(
        [belief, prior],
        ["Belief", "Prior"],
        axes.flatten(),
    ):
        ax.bar(range(len(data)), data)
        ax.set(title=title, ylim=ylim)
        ax.grid(1)


def plot_prior_posterior(prior, posterior, ylim, figwidth=8):
    _, axes = plt.subplots(
        1,
        2,
        figsize=(figwidth, figwidth / 2.5),
        constrained_layout=True,
        sharey=True,
    )
    for data, title, ax in zip(
        [prior, posterior], ["Prior", "Posterior"], axes.flatten()
    ):
        ax.bar(range(len(data)), data)
        ax.set(title=title, ylim=ylim)
        ax.grid(1)

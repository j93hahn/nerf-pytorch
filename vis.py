from fabric.utils.event import read_stats
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')


# taken from nerfy.utils
def trailing_window_view(xs, window_size):
    assert (window_size % 2) == 1, "window size should be odd"
    view = sliding_window_view(
        np.pad(xs, (window_size - 1, 0), mode="edge"), window_size
    )
    return view


def make_plot(ax, xs, psnrs, label, ws=51):
    data = trailing_window_view(psnrs, ws)
    μ = data.mean(-1)
    σ = data.std(-1)
    ax.plot(xs, μ, label=label)
    ax.fill_between(xs, μ - σ, μ + σ, alpha=0.3)
    ax.legend()


def generate_plots():
    x1, psnr1 = read_stats('experiments/no_scale_alpha/', 'psnr')
    x2, psnr2 = read_stats('experiments/scale_alpha', 'psnr')
    fig, axs = plt.subplots()
    plt.title("Experimenting with Different Scaling Factors on α for Vanilla NeRF MLP")
    plt.ylabel("PSNR Values")
    plt.xlabel("Training Iteration")
    make_plot(axs, x1, psnr1, "No Scaling on α")
    make_plot(axs, x2, psnr2, "α scaled by 30")
    plt.savefig('combinedpsnr.png', dpi=300)


if __name__ == '__main__':
    generate_plots()

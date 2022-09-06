from fabric.utils.event import read_lined_json
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')


# adapted from fabric in order to incorporate different file names
def read_stats(dirname, file, key):
    if dirname is None or not (fname := Path(dirname) / file).is_file():
        return [], []
    stats = read_lined_json(fname)
    stats = list(filter(lambda x: key in x, stats))
    xs = [e['iter'] for e in stats]
    ys = [e[key] for e in stats]
    return xs, ys


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


def main():
    x1, psnr1 = read_stats('nerf-pytorch', 'no_scaled_history.json', 'psnr0')
    x2, psnr2 = read_stats('nerf-pytorch', 'yes_scaled_history.json', 'psnr0')
    fig, axs = plt.subplots()
    plt.title("Experimenting with Different Scaling Factors on α for Vanilla NeRF MLP")
    plt.ylabel("PSNR Values")
    plt.xlabel("Training Iteration")
    make_plot(axs, x1, psnr1, "No Scaling on α")
    make_plot(axs, x2, psnr2, "α scaled by 30")
    plt.savefig('combined.png', dpi=300)


if __name__ == '__main__':
    main()

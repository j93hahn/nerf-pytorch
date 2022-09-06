from numpy.lib.stride_tricks import sliding_window_view
import numpy as np


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

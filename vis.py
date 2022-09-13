from fabric.utils.event import read_stats, read_lined_json
from utils import make_plot
from pathlib import Path

import matplotlib.pyplot as plt
import json
plt.style.use('seaborn-whitegrid')


def generate_scaled_alpha_plots():
    x1, psnr1 = read_stats('../exp/nerf/lego_train_no_scaled_alpha/', 'psnr')
    x2, psnr2 = read_stats('../exp/nerf/lego_train_scaled_alpha/', 'psnr')
    x3, psnr3 = read_stats('../exp/nerf/22_0910_1507_58/', 'psnr')
    x4, psnr4 = read_stats('../exp/nerf/22_0911_1206_52/', 'psnr')
    x5, psnr5 = read_stats('../exp/nerf/22_0912_1109_15/', 'psnr')

    #x, psnrs = read_stats('nerf-pytorch/experiments/full_res_lego_siren_pos_emb/', 'psnr')

    fig, axs = plt.subplots()
    #plt.title("Experimenting with Different Scaling Factors on α for Vanilla NeRF MLP")

    plt.title("Training PSNR Values for Vanilla NeRF on Full-Res Lego")
    plt.ylabel("PSNR Values")
    plt.xlabel("Training Iteration")
    #make_plot(axs, x, psnrs, "SiREN NeRF")

    #make_plot(axs, x1, psnr3, "BatchNorm", alpha=0.7)
    make_plot(axs, x1, psnr1, "No Scaling on α", alpha=0.7)
    make_plot(axs, x1, psnr2, "α scaled by 30", alpha=0.7)
    #make_plot(axs, x1, psnr4, "BatchNorm and 30x scaled α", alpha=0.7)
    make_plot(axs, x1, psnr5, "α divided by 30", alpha=0.7)


    plt.savefig('scaling_sigmas.png', dpi=300)


def generate_alpha_values():
    _, means = read_stats('../exp/nerf/test', 'mean')
    _, stds = read_stats('../exp/nerf/test', 'std')

    import numpy as np

    means = means[:30000]
    stds = stds[:30000]

    fig, axs = plt.subplots()
    #plt.title("Experimenting with Different Scaling Factors on α for Vanilla NeRF MLP")

    plt.title("Plotting Mean and STD of Alpha Values")
    plt.ylabel("PSNR Values")
    plt.xlabel("Training Iteration")
    make_plot(axs, np.arange(len(means)), means, "Mean")
    make_plot(axs, np.arange(len(means)), stds, "STD")
    plt.savefig("Alpha Variance Plots.png", dpi=300)


# when reading activations in, use code along the lines of:
# output_dict = [x for x in input_dict if x['type'] == '1']
# but change the keys and specific wording.
def read_stats2(dirname, key):
    if dirname is None or not (fname := Path(dirname) / "history.json").is_file():
        return [], []
    stats = read_lined_json(fname)
    stats = list(filter(lambda x: key in x, stats))
    xs = [e['iter'] for e in stats]
    ys = [e[key] for e in stats]
    return xs, ys


def read_activations():
    breakpoint()
    xs, ys = read_stats2('../exp/nerf/test', 'mean')
    breakpoint()


if __name__ == '__main__':
    generate_scaled_alpha_plots()
    #generate_alpha_values()
    #read_activations()

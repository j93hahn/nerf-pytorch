from fabric.utils.event import read_stats
from utils import make_plot

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def generate_scaled_alpha_plots():
    #x1, psnr1 = read_stats('experiments/no_scale_alpha/', 'psnr')
    #x2, psnr2 = read_stats('experiments/scale_alpha', 'psnr')

    x, psnrs = read_stats('nerf-pytorch/experiments/full_res_lego_siren_pos_emb/', 'psnr')

    fig, axs = plt.subplots()
    #plt.title("Experimenting with Different Scaling Factors on α for Vanilla NeRF MLP")

    plt.title("Training PSNR Values for SiREN NeRF on Lego")
    plt.ylabel("PSNR Values")
    plt.xlabel("Training Iteration")
    make_plot(axs, x, psnrs, "SiREN NeRF")

    #make_plot(axs, x1, psnr1, "No Scaling on α")
    #make_plot(axs, x2, psnr2, "α scaled by 30")


    plt.savefig('siren_full_res.png', dpi=300)


if __name__ == '__main__':
    generate_scaled_alpha_plots()

from fabric.utils.event import read_stats
import matplotlib.pyplot as plt


def main():
    xs, ys = read_stats('nerf-pytorch', 'psnr0')
    #plt.plot(ys)
    #plt.savefig('yes.png', dpi=300)


if __name__ == '__main__':
    main()

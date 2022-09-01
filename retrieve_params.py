import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np


plt.style.use('seaborn-whitegrid')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Load pretrained NeRF model')
    parser.add_argument(
        '-o',
        default='lego',
        choices=['fern', 'flower', 'fortress', 'horns', 'lego', 'trex'],
        help='Specify the model you would you like to retrieve')

    args = vars(parser.parse_args())

    number = '2' if args['o'] in ['fern', 'lego'] else '1'
    return 'nerf_pytorch/logs/' + args['o'] + '_test/' + number + '00000.tar'


"""
network_fn is used to sample 64 "fine" points, while network_fine samples those
64 "fine" points plus 128 "coarse" points = 192 total points. in the case
network_fine is None, then network_fn is used.
"""
def main():
    location = parse_args()
    ckpt = torch.load(location)
    network_fn = ckpt['network_fn_state_dict']
    network_fine = ckpt['network_fine_state_dict']

    #print("\nPrinting network_fn parameter groups\n")
    for key in network_fn.keys():
        if key == 'alpha_linear.weight':
            #breakpoint()
            print(network_fn[key].max())
            x = np.histogram(network_fn[key].reshape(-1).cpu().numpy(), density=True)
            plt.plot(x[0])
            plt.savefig('first_view.png', dpi=300)

    #print("\nPrinting network_fine parameter groups\n")
    #for key in network_fine.keys():
    #    print(key, network_fine[key].shape)

    return


if __name__ == '__main__':
    main()

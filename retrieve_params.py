import matplotlib.pyplot as plt
import configargparse
import torch
import numpy as np


def config_parser():
    parser = configargparse.ArgumentParser(
        description='Load pretrained NeRF model')
    parser.add_argument(
        '--object',
        default='lego',
        choices=['fern', 'flower', 'fortress', 'horns', 'lego', 'trex'],
        help='Specify the model you would you like to retrieve')
    parser.add_argument(
        '--model',
        default='fn',
        choices=['fn', 'fine'],
        help='Specify the state dictionary to retrieve')

    return parser


# network_fn is used to sample 64 "fine" points, while network_fine samples those
# 64 "fine" points plus 128 "coarse" points = 192 total points. in the case
# network_fine is None, then network_fn is used.
def process_args(args):
    params = 'network_' + args.model + '_state_dict'
    number = '2' if args.object in ['fern', 'lego'] else '1'
    loc = 'nerf-pytorch/logs/' + args.object + '_test/' + number + '00000.tar'
    return params, loc


def main():
    parser = config_parser()
    args = parser.parse_args()
    params, loc = process_args(args)

    ckpt = torch.load(loc)
    network_weights = ckpt[params]

    #print("\nPrinting network_fn parameter groups\n")
    for key in network_weights.keys():
        if key == 'alpha_linear.weight':
            #breakpoint()
            print(network_weights[key].max())
            x = np.histogram(network_weights[key].reshape(-1).cpu().numpy(), density=True)
            plt.plot(x[0])
            plt.savefig('first_view.png', dpi=300)


if __name__ == '__main__':
    plt.style.use('seaborn-whitegrid')
    main()

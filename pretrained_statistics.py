import configargparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_parser():
    parser = configargparse.ArgumentParser(
        description='Load pretrained NeRF model and retrieve parameter statistics')
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
    state = 'network_' + args.model + '_state_dict'
    number = '2' if args.object in ['fern', 'lego'] else '1'
    loc = 'nerf-pytorch/pretrained/' + args.object + '_test/' + number + '00000.tar'
    return state, loc


def main():
    parser = config_parser()
    args = parser.parse_args()
    state, loc = process_args(args)

    # ckpt contains four keys, of which we are only interested in the two that
    # contain the model state dictionaries
    ckpt = torch.load(loc, map_location=device)
    params = ckpt[state]

    # note: each key stores only one set of weights or one set of biases for a layer
    # it does not store all of a layer's params
    for key in params.keys():
        x = params[key]
        print(key)
        print("Shape:", x.shape)
        print("Mean:", x.mean().item())
        print("Std.:", x.std().item())
        print("")
        #if key == 'alpha_linear.weight':
        #    x = params[key].reshape(-1).cpu().numpy()
        #    y = np.histogram(x, density=True)
        #    plt.hist(y[0], bins=y[1].shape[0])
        #    plt.savefig('first_view4.png', dpi=300)
        #    print(x.mean(), x.std())


if __name__ == '__main__':
    main()

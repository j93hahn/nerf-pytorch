from fabric.utils.event import read_lined_json
from utils import make_plot
from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import os


total = 30000 # number of datapoints per rendered pose - 12 * 2500
plt.style.use('seaborn-whitegrid')


def read_stats(dirname):
    if dirname is None or not (fname := Path(dirname) / "history.json").is_file():
        return [], []
    stats = read_lined_json(fname)
    return list(stats)


# in each 12 points, stats[0:8] contain the standard 8-layer MLP. stats[8] contains
# the alpha values, stats[9] has feature values, stats[10] has the "9th" layer,
# stats[11] outputs the RGB values (before sigmoid activation)
def retrieve(rendered_pose:int, layer:Union[int, str], stat:str):
    stats = read_stats(dirname='../exp/nerf/lego_train_scaled_alpha/test/render_pose' + str(rendered_pose))
    data = []

    def preprocessing(layer):
        valid_str = ['alpha', 'feature', 'layer', 'rgb']
        discriminator = None

        if layer in range(8):
            discriminator = layer
        elif layer in valid_str:
            discriminator = valid_str.index(layer) + 8
        else:
            raise Exception("Invalid layer input")

        return discriminator

    discriminator = preprocessing(layer)

    for i in range(total):
        # assertion passes -> data logged correctly
        # if i % 12 in range(8):
        #     assert stats[i]['layer'] == i % 12
        if i % 12 == discriminator:
            data.append(stats[i][stat])

    return np.array(data)


def vis(data, rendered_pose):
    plt.hist(data[:, 0], bins=50, alpha=0.5, label="Means")
    plt.hist(data[:, 1], bins=50, alpha=0.5, label="Stds")
    plt.title("Activation Distribution Layer 9 for Pretrained Lego Rendered Pose " + str(rendered_pose))
    plt.xlabel("Values with Scaled Sigmas")
    plt.ylabel("Count")
    plt.legend()

    os.chdir('../exp/nerf/lego_train_scaled_alpha/test/render_pose' + str(rendered_pose))

    plt.savefig('activation_distrib_layer9.png', dpi=300)


if __name__ == '__main__':
    rendered_pose = 1
    layer = 'layer'

    means = retrieve(rendered_pose, layer, 'mean')
    stds = retrieve(rendered_pose, layer, 'std')

    data = np.stack([means, stds], axis=-1)

    vis(data, rendered_pose)

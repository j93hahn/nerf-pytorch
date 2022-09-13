from fabric.utils.event import read_lined_json
from utils import make_plot
from typing import Union
from pathlib import Path
from tqdm import tqdm

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
def retrieve(pose:int, layer:Union[int, str], stat:str):
    stats = read_stats(dirname='./')
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


def vis(data, pose, element):
    def element_to_title(element):
        if element in range(8):
            return f"Layer {element+1} Activation"
        elif element == 'alpha':
            return "Sigma"
        elif element == 'rgb':
            return "RGB"
        elif element == 'feature':
            return "'Feature'"
        elif element == 'layer':
            return "Layer 9 Activation"
        else:
            raise Exception("Invalid element input")

    title = element_to_title(element)

    plt.hist(data[:, 0], bins=50, alpha=0.5, label="Means")
    plt.hist(data[:, 1], bins=50, alpha=0.5, label="Stds")

    plt.title(f"{title} Distribution for Pretrained Lego Rendered Pose " + str(pose+1))
    plt.xlabel("Values")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(f'{title} Distribution.png', dpi=300)
    plt.clf()


if __name__ == '__main__':
    pose = 0
    for element in tqdm([0, 1, 2, 3, 4, 5, 6, 7, 'layer', 'feature', 'rgb', 'alpha']):
        means = retrieve(pose, element, 'mean')
        stds = retrieve(pose, element, 'std')
        data = np.stack([means, stds], axis=-1)
        vis(data, pose, element)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from torchviz import make_dot


# this module serves the purpose of Positional Embedding -- rather than positionally
# embedding the inputs to the SIREN MLP, we replace it with a linear layer and run it
# through a Sine activation function

# in the case of embedding the input coordinates, we have out_features=63 and
# for viewing directions, out_features=27
class SirenEmbedding(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=bias)
        self.omega_0 = omega_0
        SirenEmbedding._first_layer_sine_init(self.layer)

    @staticmethod
    def _first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.layer(x))


# this bundles the Linear layer and initialization technique together - not used!
# sourced from https://github.com/kwea123/Coordinate-MLPs/blob/master/models.py
class SirenLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.in_features) / self.omega_0,
                                             np.sqrt(6/self.in_features) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# NeRF model implementation with MLP backbone replaced by SiREN
class Siren(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(Siren, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # create layers here
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self._init_weights()

    # apply weight initialization here - create the model first then initialize weights
    def _init_weights(self):
        for i in range(len(self.pts_linears)):
            Siren._sine_init(self.pts_linears[i])

        Siren._sine_init(self.views_linears)

        if self.use_viewdirs:
            Siren._sine_init(self.feature_linear)
            Siren._sine_init(self.alpha_linear)
            Siren._sine_init(self.rgb_linear)
        else:
            Siren._sine_init(self.output_linear)

    # the following two static methods were taken directly from SIREN
    @staticmethod
    def _first_layer_sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1 / num_input, 1 / num_input)

    @staticmethod
    def _sine_init(m):
        with torch.no_grad():
            if hasattr(m, 'weight'):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

    # sine activation function - replaces ReLU in NeRF MLP, taken directly from SIREN
    @staticmethod
    def Sine(x):
        return torch.sin(30 * x)

    # forward pass identical to NeRF -- replace ReLU activation with Sine
    def forward(self, x):
        breakpoint()
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = Siren.Sine(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = Siren.Sine(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        #make_dot(outputs).render("attached", format="png")
        return outputs


def compare_models():
    model2 = Siren()


if __name__ == '__main__':
    compare_models()

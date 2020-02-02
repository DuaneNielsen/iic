import torch
from torch import nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def initialize_weights(f):
    for m in f.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def load_weights(module, weights):
    start = 0
    end = 0
    for p in module.parameters():
        end += p.numel()
        p.data = weights[start:end].reshape(p.shape)
        start += p.numel()
    return module


def flatten(module):
    return torch.cat([p.flatten() for p in module.parameters()])


def parameter_count(module):
    return sum([p.numel() for p in module.parameters()])


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is int:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


"""
M -> MaxPooling
L -> Capture Activations for Perceptual loss
U -> Bilinear upsample
"""


def make_layers(cfg,
                type='conv',
                batch_norm=True,
                nonlinearity=None,
                nonlinearity_kwargs=None,
                input_shape=None):

    nonlinearity_kwargs = {} if nonlinearity_kwargs is None else nonlinearity_kwargs
    nonlinearity = nn.ReLU(inplace=True) if nonlinearity is None else nonlinearity(**nonlinearity_kwargs)

    output_shape = input_shape
    output_channels = None

    layers = []
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            output_shape = conv_output_shape(output_shape, kernel_size=2, stride=2)
        elif v == 'U':
            layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
            output_shape = [2 * i for i in output_shape]
        else:
            if type == 'conv':
                layers += [nn.ReplicationPad2d(1)]
                layers += [nn.Conv2d(in_channels, v, kernel_size=3)]
            if type == 'fc':
                layers += [nn.Linear(in_channels, v)]
            if batch_norm:
                if type == 'conv':
                    layers += [nn.BatchNorm2d(v)]
                if type == 'fc':
                    layers += [nn.BatchNorm1d(v)]
            layers += [nonlinearity]

            in_channels = v
            output_channels = v

    if input_shape is not None:
        return nn.Sequential(*layers), (output_channels, *output_shape)
    else:
        return nn.Sequential(*layers)

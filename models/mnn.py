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


"""
M -> MaxPooling
L -> Capture Activations for Perceptual loss
U -> Bilinear upsample
"""


def make_layers(cfg,
                type='conv',
                batch_norm=True,
                nonlinearity=None,
                nonlinearity_kwargs=None):

    nonlinearity_kwargs = {} if nonlinearity_kwargs is None else nonlinearity_kwargs
    nonlinearity = nn.ReLU(inplace=True) if nonlinearity is None else nonlinearity(**nonlinearity_kwargs)

    layers = []
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'U':
            layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
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
    return nn.Sequential(*layers)

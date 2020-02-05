import torch
from torch import nn as nn
from models.layerbuilder import FCBuilder
from models.resnet import ResNetBuilder, ResNetFixupBuilder, StableResNetBuilder, ConstResNetFixupBuilder
from models.vgg import VGGNetBuilder


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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


builders = {'vgg': VGGNetBuilder(),
            'fc': FCBuilder(),
            'resnet': StableResNetBuilder(),
            'resnet-fixup': ResNetFixupBuilder(),
            'resnet-fixup-const': ConstResNetFixupBuilder(),
            }


def make_layers(cfg, type, input_shape, **kwargs):
    return builders[type].make_layers(cfg, input_shape, **kwargs)


def __make_layers(cfg,
                type='conv',
                nonlinearity=None,
                nonlinearity_kwargs=None,
                input_shape=None):

    nonlinearity_kwargs = {} if nonlinearity_kwargs is None else nonlinearity_kwargs
    nonlinearity = nn.ReLU(inplace=True) if nonlinearity is None else nonlinearity(**nonlinearity_kwargs)

    output_shape = input_shape[1:3]
    output_channels = None

    layers = []
    in_channels = cfg[0]
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            import models
            if input_shape is not None:
                output_shape = models.layerbuilder.conv_output_shape(output_shape, kernel_size=2, stride=2)
                if min(*output_shape) <= 0:
                    raise Exception('Image downsampled to 0 or less, use less downsampling')

        elif v == 'U':
            layers += [nn.UpsamplingBilinear2d(scale_factor=2)]

            if input_shape is not None:
                output_shape = [2 * i for i in output_shape]

        else:
            if type == 'vgg':
                layers += [nn.ReplicationPad2d(1)]
                layers += [nn.Conv2d(in_channels, v, kernel_size=3)]
                output_channels = v
            if type == 'fc':
                layers += [nn.Linear(in_channels, v)]
                output_channels = v

            layers += [nonlinearity]

            in_channels = v

    if input_shape is not None:
        if type == 'fc':
            output_shape = (output_channels,)
        if type == 'vgg':
            output_shape = (output_channels, *output_shape)

        return nn.Sequential(*layers), output_shape
    else:
        return nn.Sequential(*layers)

import torch
from torch import nn as nn
from models.layerbuilder import FCBuilder, LayerMetaData
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


builders = {'vgg16': VGGNetBuilder(),
            'fc': FCBuilder(),
            'resnet-batchnorm': StableResNetBuilder(),
            'resnet-fixup': ResNetFixupBuilder(),
            'resnet-fixup-const': ConstResNetFixupBuilder(),
            'resnet': ConstResNetFixupBuilder(),
            'resnet-conv-shortcut': ResNetBuilder()
            }


def make_layers(cfg, type, meta, **kwargs):
    return builders[type].make_layers(cfg, meta, **kwargs)
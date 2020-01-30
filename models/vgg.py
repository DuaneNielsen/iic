import torch.nn as nn
import models.mnn as knn
from math import floor

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


decoder_cfg = {
    'A': [512, 512, 'U', 256, 256, 'U', 256, 256, 'U', 128, 'U', 64, 'U'],
    'F': [512, 512, 'U', 256, 256, 'U', 256, 256, 'U', 128, 64],
    'G': [512, 512, 'U', 256, 256, 'U', 128, 64],
    'VGG_PONG': [32, 'U', 16, 'U', 16],
    'VGG_PONG_TRIVIAL': [16, 16],
    'VGG_PONG_LAYERNECK': [32, 32, 16, 16],
    'VGG_PACMAN': [16, 32, 32, 16],
    'VGG_PACMAN_2': [64, 'U', 32, 32, 16],
}

vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'G': [64, 128, 'M', 256, 256, 'M', 512, 512],
    'VGG_PONG': [16, 'M', 16, 'M', 32],
    'VGG_PONG_TRIVIAL': [16, 16],
    'VGG_PONG_LAYERNECK': [16, 32],
    'VGG_PACMAN': [16, 32, 32, 16],
    'VGG_PACMAN_2': [16, 32, 32, 'M', 64],
    'MAPPER': [8, 8],
}

from torch import nn as nn

from models.layerbuilder import LayerBuilder


class VGGNetBuilder(LayerBuilder):
    def __init__(self, ):
        super().__init__()

    def make_block(self, in_channels, v):
        self.layers += [nn.ReplicationPad2d(1)]
        self.layers += [nn.Conv2d(in_channels, v, kernel_size=3)]
        self.layers += [nn.BatchNorm2d(v)]
        self.layers += [self.nonlinearity]
        self.shape = (v, self.shape[1], self.shape[2])
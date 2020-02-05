from torch import nn as nn

from models.layerbuilder import LayerBuilder


class VGGNetBuilder(LayerBuilder):
    def __init__(self, ):
        super().__init__()

    @staticmethod
    def initialize_weights(f):
        for m in f.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_block(self, in_channels, v):
        self.layers += [nn.ReplicationPad2d(1)]
        self.layers += [nn.Conv2d(in_channels, v, kernel_size=3)]
        self.layers += [nn.BatchNorm2d(v)]
        self.layers += [self.nonlinearity]
        self.meta.shape = (v, self.meta.shape[1], self.meta.shape[2])
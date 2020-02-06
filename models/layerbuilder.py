import torch.nn as nn
from torch import nn as nn


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


class LayerMetaData:
    def __init__(self, input_shape):
        self.shape = input_shape
        self.depth = 1


"""
M -> MaxPooling
L -> Capture Activations for Perceptual loss
U -> Bilinear upsample
"""

def scan_token(token):
    if type(token) is str:
        t = token.split(':')
        if len(t) == 2:
            return t[0], int(t[1])
        else:
            return t[0], 0
    elif type(token) is int:
        return '', token


class LayerBuilder:
    def __init__(self):
        self.layers = []
        self.meta = None
        self.output_channels = None
        self.nonlinearity = None

    # called by make_layers to initialize variables
    def new_layer_hook(self):
        pass

    @staticmethod
    def initialize_weights(f):
        pass

    def make_block(self, in_channels, v):
        pass

    def make_layers(self, cfg, meta, nonlinearity=None, nonlinearity_kwargs=None):
        self.layers = []
        self.output_channels = nonlinearity_kwargs
        self.nonlinearity = nonlinearity
        self.new_layer_hook()
        self.meta = meta

        nonlinearity_kwargs = {} if nonlinearity_kwargs is None else nonlinearity_kwargs
        self.nonlinearity = nn.ReLU(inplace=True) if nonlinearity is None else nonlinearity(**nonlinearity_kwargs)

        tipe, in_channels = scan_token(cfg[0])
        for token in cfg[1:]:
            tipe, value = scan_token(token)
            if tipe == 'M':
                self.layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                self.meta.shape = (self.meta.shape[0], *conv_output_shape(self.meta.shape[1:3], kernel_size=2, stride=2))
                if min(*self.meta.shape[1:3]) <= 0:
                    raise Exception('Image downsampled to 0 or less, use less downsampling')

            elif tipe == 'U':
                self.layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
                self.meta.shape = (self.meta.shape[0], self.meta.shape[1] * 2, self.meta.shape[2] * 2)

            elif tipe == 'C':
                self.layers += [nn.Conv2d(in_channels, value, kernel_size=3, stride=1, padding=1, bias=False)]
                self.layers += [self.nonlinearity]
                self.meta.depth += 1
                self.meta.shape = value, *conv_output_shape(self.meta.shape[1:3], kernel_size=3, stride=1, pad=1)

            elif tipe == '':
                self.make_block(in_channels, value)
                in_channels = value

        layer = nn.Sequential(*self.layers)
        self.initialize_weights(layer)
        return layer, self.meta


class FCBuilder(LayerBuilder):

    @staticmethod
    def initialize_weights(f):
        for m in f.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_block(self, in_channels, v):
        self.layers += [nn.Linear(in_channels, v)]
        self.layers += [nn.BatchNorm1d(v)]
        self.layers += [self.nonlinearity]
        self.meta.shape = (v,)
import torch.nn as nn
import torch.nn.functional as F
from models.layerbuilder import LayerBuilder


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.p1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.p2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.p1(x))))
        out = self.bn2(self.conv2(self.p2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetBuilder(LayerBuilder):
    def __init__(self):
        super().__init__()

    def make_block(self, in_channels, v):
        self.layers += [BasicBlock(in_channels, v)]
        self.shape = (v, self.shape[1], self.shape[2])


# def make_layers(cfg, input_shape):
#     in_planes = cfg[0]
#     layers = []
#     output_shape = input_shape
#
#     for v in cfg[1:]:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#
#             if input_shape is not None:
#                 output_shape = mnn.conv_output_shape(output_shape, kernel_size=2, stride=2)
#                 if min(*output_shape) <= 0:
#                     raise Exception('Image downsampled to 0 or less, use less downsampling')
#
#         elif v == 'U':
#             layers += [nn.UpsamplingBilinear2d(scale_factor=2)]
#
#             if input_shape is not None:
#                 output_shape = [2 * i for i in output_shape]
#
#         else:
#             layers += [BasicBlock(in_planes, v)]
#             in_planes = v
#
#     return nn.Sequential(*layers), (in_planes, *output_shape)

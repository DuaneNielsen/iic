import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layerbuilder import LayerBuilder, conv_output_shape
from torch.nn.functional import avg_pool2d

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
        self.layers += [BasicBlock(in_channels, v)]
        self.shape = (v, self.shape[1], self.shape[2])


class FixupResLayer(nn.Module):
    def __init__(self, depth, in_layers, filters, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_layers, filters, 3, stride=stride, padding=1, bias=False)
        self.c1.weight.data.mul_(depth ** -0.5)
        self.c2 = nn.Conv2d(filters, filters, 3, stride=1, padding=1, bias=False)
        self.c2.weight.data.zero_()
        self.stride = stride

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(4)])

    def forward(self, input):
        hidden = input + self.bias[0]
        hidden = self.c1(hidden) + self.bias[1]
        hidden = torch.relu(hidden) + self.bias[2]
        hidden = self.c2(hidden) * self.gain + self.bias[3]

        # pad the image if its size is not divisible by 2
        padding_h = 0 if input.size(2) % 2 == 0 else 1
        padding_w = 0 if input.size(3) % 2 == 0 else 1
        id = avg_pool2d(input, self.stride, stride=self.stride, padding=(padding_h, padding_w))

        # this assumes we are always doubling the amount of kernels as we go deeper
        if id.size(1) != hidden.size(1):
            id = torch.cat((id, id), dim=1)
        return torch.relu(hidden + id)


class DeepResNetFixup(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(3, 64, 3, bias=False)
        self.layer1 = nn.Sequential(*[FixupResLayer(2, 64, 64), FixupResLayer(3, 64, 64)])
        self.layer2 = nn.Sequential(*[FixupResLayer(4, 64, 128, stride=2), FixupResLayer(5, 128, 128, stride=2)])
        self.layer3 = nn.Sequential(*[FixupResLayer(6, 128, 256, stride=2), FixupResLayer(7, 256, 256, stride=2)])
        self.layer4 = nn.Sequential(*[FixupResLayer(8, 256, 512, stride=2), FixupResLayer(9, 512, 512, stride=2)])
        self.pool = nn.AvgPool2d(4, padding=1)
        self.out = nn.Linear(512, 10)
        self.out.weight.data.zero_()
        self.out.bias.data.zero_()

    def forward(self, input):
        hidden = torch.relu(self.first(input))
        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        hidden = self.layer4(hidden)
        #hidden = self.pool(hidden).squeeze()
        return self.out(hidden.squeeze())


class ResNetFixupBuilder(LayerBuilder):
    """
    Note that this assumes the inputs are normalized.  If you use the output of a Fixup network
    directly into another fixup, you will have to carry the depth over.

    """
    def __init__(self):
        super().__init__()
        self.depth = 1

    @staticmethod
    def initialize_weights(f):
        pass

    def make_block(self, in_channels, v):
        if self.depth == 1:
            self.layers += [nn.Conv2d(in_channels, v, kernel_size=3, bias=False)]
            self.layers += [self.nonlinearity]
            self.depth += 1
            self.shape = v, *conv_output_shape(self.shape[1:3], kernel_size=3, stride=3)
        else:
            self.layers += [FixupResLayer(self.depth, in_channels, v, stride=2)]
            self.depth += 1
            self.shape = v, *conv_output_shape(self.shape[1:3], kernel_size=3, stride=2, pad=1)



# class FixupBlock(nn.Module):
#     def __init__(self, in_planes, planes, depth, stride=2):
#         super(FixupBlock, self).__init__()
#         self.stride = stride
#         #self.p1 = nn.ZeroPad2d(1)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         #nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
#         self.conv1.weight.data.mul_(depth ** -0.5)
#         #self.p2 = nn.ZeroPad2d(1)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv2.weight.data.zero_()
#         self.gain = nn.Parameter(torch.ones(1), requires_grad=True)
#         self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(4)])
#
#     def forward(self, x):
#         out = x + self.bias[0]
#         #out = self.conv1(self.p1(out)) + self.bias[1]
#         out = self.conv1(out) + self.bias[1]
#         out = F.relu(out) + self.bias[2]
#         #out = self.conv2(self.p2(out)) * self.gain + self.bias[3]
#         out = self.conv2(out) * self.gain + self.bias[3]
#
#         padding_h = 0 if x.size(2) % 2 == 0 else 1
#         padding_w = 0 if x.size(3) % 2 == 0 else 1
#         id = avg_pool2d(x, self.stride, stride=self.stride, padding=(padding_h, padding_w))        # pad the image if its size is not divisible by 2
#         # this assumes we are always doubling the amount of kernels as we go deeper
#         if id.size(1) != out.size(1):
#             id = torch.cat((id, id), dim=1)
#         out = F.relu(out + id)
#         return out
#
#
# class ResNetFixupBuilder(LayerBuilder):
#     """
#     Note that this assumes the inputs are normalized.  If you use the output of a Fixup network
#     directly into another fixup, you will have to carry the depth over.
#
#     """
#     def __init__(self, depth=1):
#         super().__init__()
#         self.depth = depth
#
#     @staticmethod
#     def initialize_weights(f):
#         pass
#
#     def make_block(self, in_channels, v):
#
#         if self.depth == 1:
#             self.layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False)]
#             self.layers += [self.nonlinearity]
#             self.shape = (v, self.shape[1], self.shape[2])
#             self.depth += 1
#         else:
#             self.layers += [FixupBlock(in_channels, v, self.depth)]
#             self.shape = (v, *conv_output_shape(self.shape[1:3], kernel_size=3, stride=2, pad=1))
#             self.depth += 1


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

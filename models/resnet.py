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
        self.meta.shape = (v, self.meta.shape[1], self.meta.shape[2])


class StableBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(StableBlock, self).__init__()
        self.p1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.p2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(self.p1(x))))
        out = self.bn2(self.conv2(self.p2(out)))

        # pad the image if its size is not divisible by 2
        padding_h = 0 if x.size(2) % 2 == 0 else 1
        padding_w = 0 if x.size(3) % 2 == 0 else 1
        id = avg_pool2d(x, 1, stride=1, padding=(padding_h, padding_w))

        # this assumes we are always doubling the amount of kernels as we go deeper
        if id.size(1) != out.size(1):
            id = torch.cat((id, id), dim=1)

        out = F.relu(out + id)
        return out


class StableResNetBuilder(LayerBuilder):
    def __init__(self):
        super().__init__()

    @staticmethod
    def initialize_weights(f):
        for m in f.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_block(self, in_channels, v):
        if self.meta.depth == 1:
            self.layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False)]
            self.layers += [nn.BatchNorm2d(v)]
            self.layers += [self.nonlinearity]
            self.meta.depth += 1
            self.meta.shape = v, *conv_output_shape(self.meta.shape[1:3], kernel_size=3, stride=1, pad=1)
        else:
            self.layers += [StableBlock(in_channels, v)]
            self.meta.shape = v, *conv_output_shape(self.meta.shape[1:3], kernel_size=3, stride=1, pad=1)
            self.meta.depth += 1


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

        # if more channels in the next layer, then double
        if id.size(1) < hidden.size(1):
            id = torch.cat((id, id), dim=1)

        # if less channels in next layer, then halve
        if id.size(1) > hidden.size(1):
            id = torch.add(*id.chunk(2, dim=1)) / 2.0

        return torch.relu(hidden + id)


class ResNetFixupBuilder(LayerBuilder):
    """
    Note that this assumes the inputs are normalized.  If you use the output of a Fixup network
    directly into another fixup, you will have to carry the depth over.

    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def initialize_weights(f):
        """  Null init, blocks init themselves"""
        pass

    def make_block(self, in_channels, v):
        if self.meta.depth == 1:
            self.layers += [nn.Conv2d(in_channels, v, kernel_size=3, bias=False)]
            self.layers += [self.nonlinearity]
            self.meta.depth += 1
            self.meta.shape = v, *conv_output_shape(self.meta.shape[1:3], kernel_size=3, stride=3)
        else:
            self.layers += [FixupResLayer(self.meta.depth, in_channels, v, stride=2)]
            self.meta.depth += 1
            self.meta.shape = v, *conv_output_shape(self.meta.shape[1:3], kernel_size=3, stride=2, pad=1)


class ConstFixupResLayer(nn.Module):
    def __init__(self, depth, in_layers, filters, stride=1):
        """ Same HW out as in"""
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


class ConstResNetFixupBuilder(LayerBuilder):
    """
    Note that this assumes the inputs are normalized.  If you use the output of a Fixup network
    directly into another fixup, you will have to carry the depth over.

    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def initialize_weights(f):
        """  Null init, blocks init themselves"""
        pass

    def make_block(self, in_channels, v):
        if self.meta.depth == 1:
            self.layers += [nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1, bias=False)]
            self.layers += [self.nonlinearity]
            self.meta.depth += 1
            self.meta.shape = v, *conv_output_shape(self.meta.shape[1:3], kernel_size=3, stride=1, pad=1)
        else:
            self.layers += [FixupResLayer(self.meta.depth, in_channels, v, stride=1)]
            self.meta.depth += 1
            self.meta.shape = v, *conv_output_shape(self.meta.shape[1:3], kernel_size=3, stride=1, pad=1)
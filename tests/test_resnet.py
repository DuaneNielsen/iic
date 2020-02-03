import torch
import models.resnet as rn


def test_make_layers():
    conf = [3, 16, 32, 'M']
    x = torch.randn((10, 3, 12, 12))
    builder = rn.ResNetBuilder()
    resnet, output_shape = builder.make_layers(conf, x[0].shape)
    y = resnet(x)
    assert output_shape ==(32, 6, 6)
    assert y.shape[1:] == (32, 6, 6)

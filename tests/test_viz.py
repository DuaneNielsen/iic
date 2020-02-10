import torch

from train_classifier import show, Guesser
from utils.viewer import UniImageViewer
from torchvision.datasets import MNIST
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

viewer = UniImageViewer()


def test_show():
    x = torch.randn(32, 1, 28, 28)
    y = torch.randn(32, 10)
    viewer.render(show(x, y), block=True)


def test_with_mnist():
    data = MNIST(root='../data', train=True, transform=ToTensor())
    loader = DataLoader(data, 10)
    x, y = next(loader.__iter__())
    y = one_hot(y, 10)
    viewer.render(show(x, y), block=True)


def test_guess_class():
    y = one_hot(torch.tensor([0, 1, 2, 3, 4, 5, 6, 9, 8, 9]))
    target = torch.tensor([0, 1, 2, 3, 4, 5, 6, 9, 8, 9])
    g = Guesser(10)
    g.add(y, target)
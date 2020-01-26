from pathlib import Path

import torch
import torchvision as tv
from torch import randperm
from torch._utils import _accumulate
import datasets.celeba


def random_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) > len(dataset):
        raise ValueError("Sum of input lengths is greater than the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in
            zip(_accumulate(lengths), lengths)]


def split(data, train_len, test_len):
    total_len = train_len + test_len
    train = torch.utils.data.Subset(data, range(0, train_len))
    test = torch.utils.data.Subset(data, range(train_len, total_len))
    return train, test


class DataPack(object):
    def __init__(self):
        self.name = None
        self.transforms = None

    def make(self, train_len, test_len, **kwargs):
        pass


class ImageDataPack(DataPack):
    def __init__(self, name, subdir, transforms):
        super().__init__()
        self.name = name
        self.transforms = transforms
        self.subdir = subdir

    def make(self, train_len, test_len, data_root='data', **kwargs):
        """
        Returns a test and training dataset
        :param train_len: images in training set
        :param test_len: images in test set
        :param data_root: the the root directory for project datasets
        :return:
        """

        data = tv.datasets.ImageFolder(str(Path(data_root) / Path(self.subdir)), transform=self.transforms, **kwargs)
        return split(data, train_len, test_len)


class Builtin(DataPack):
    def __init__(self, clazz, transform):
        super().__init__()
        self.transforms = transform
        self.clazz = clazz

    def make(self, train_len, test_len, data_root='data', **kwargs):
        train = self.clazz(data_root, train=True, transform=self.transforms, download=True)
        test = self.clazz(data_root, train=False, transform=self.transforms, download=True)
        return train, test


datasets = {
    'celeba': ImageDataPack('celeba', 'celeba-low', datasets.celeba.celeba_transform),
    'cifar-10': Builtin(tv.datasets.CIFAR10, tv.transforms.ToTensor()),
    'mnist': Builtin(tv.datasets.MNIST, tv.transforms.ToTensor())
}
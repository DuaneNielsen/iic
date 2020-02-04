from pathlib import Path

import torch
import torchvision as tv
from torch import randperm
from torch._utils import _accumulate
import datasets.celeba
import datasets.cifar10
import datasets.mnist
from torch.utils.data import Subset


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
    train = Subset(data, range(0, train_len))
    test = Subset(data, range(train_len, total_len))
    return train, test


class DataPack(object):
    def __init__(self):
        self.name = None
        self.train_transform = None
        self.test_transform = None
        self.class_list = []
        self.hw = None

    def make(self, train_len, test_len, **kwargs):
        pass

    def add_empty_columns(self, n):
        self.columns = list(range(n))

    def add_classes(self, class_list, class_n):
        self.class_list = class_list
        if class_list is None and class_n is not None:
            self.class_list = list(range(class_n))

    @property
    def num_classes(self):
        return len(self.class_list)


class ImageDataPack(DataPack):
    def __init__(self, name, subdir, train_transform, test_transform, class_list=None, class_n=None):
        super().__init__()
        self.name = name
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.subdir = subdir
        self.add_classes(class_list, class_n)

    def make(self, train_len, test_len, data_root='data', classes=None, classes_n=None, **kwargs):
        """
        Returns a test and training dataset
        :param train_len: images in training set
        :param test_len: images in test set
        :param data_root: the the root directory for project datasets
        :return:
        """

        data = tv.datasets.ImageFolder(str(Path(data_root) / Path(self.subdir)), transform=self.transforms, **kwargs)
        self.shape = data[0][0].shape
        return split(data, train_len, test_len)


class Builtin(DataPack):
    def __init__(self, torchv_class, train_transform, test_transform, class_list=None, class_n=None):
        super().__init__()
        self.train_transform = train_transform
        self.test_transform = test_transform

        self.torchv_class = torchv_class
        self.add_classes(class_list, class_n)

    def make(self, train_len, test_len, data_root='data', **kwargs):
        train = self.torchv_class(data_root, train=True, transform=self.train_transform, download=True)
        test = self.torchv_class(data_root, train=False, transform=self.test_transform, download=True)
        self.shape = train[0][0].shape
        if train_len is not None:
            train_len = min(train_len, len(train))
            train = Subset(train, range(0, train_len))
        if test_len is not None:
            test_len = min(test_len, len(test))
            test = Subset(test, range(0, test_len))
        return train, test


datasets = {
    'celeba': ImageDataPack('celeba', 'celeba-low', datasets.celeba.celeba_transform, datasets.celeba.celeba_transform),
    'cifar-10-no-aug': Builtin(tv.datasets.CIFAR10,
                               tv.transforms.ToTensor(),
                               tv.transforms.ToTensor(),
                               class_list=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                           'truck']),
    'cifar-10': Builtin(tv.datasets.CIFAR10,
                        train_transform=datasets.cifar10.transform_train,
                        test_transform=datasets.cifar10.transform_test,
                        class_list=['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
                                    'truck']),
    'mnist': Builtin(tv.datasets.MNIST, datasets.mnist.train_transform, datasets.mnist.test_transform, class_n=10)
}

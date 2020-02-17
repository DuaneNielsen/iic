from pathlib import Path

import torch
import torchvision as tv
from torch import randperm
from torch._utils import _accumulate
import datasets.celeba
import datasets.cifar10
import datasets.mnist
from torch.utils.data import Subset
from torchvision.transforms import ToTensor


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
        self.train_augment = None
        self.test_augment = None
        self.class_list = []
        self.hw = None

    def make(self, train_len, test_len, **kwargs):
        pass

    def add_empty_columns(self, n):
        self.columns = list(range(n))

    def add_classes(self, class_list, class_n):
        self.class_list = class_list
        if class_list is None and class_n is not None:
            self.class_list = [str(i) for i in range(class_n)]

    @property
    def num_classes(self):
        return len(self.class_list)


class TransformDataset:
    def __init__(self, dataset, baseline_tranform=None, augment_transform=None):
        if baseline_tranform:
            self.baseline_transform = baseline_tranform
        else:
            self.baseline_transform = ToTensor()
        self.augment_transform = augment_transform
        self.dataset = dataset

    def __getitem__(self, item):
        x, target = self.dataset[item]
        return self.baseline_transform(x), self.augment_transform(x), target

    def __len__(self):
        return len(self.dataset)


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

        data = tv.datasets.ImageFolder(str(Path(data_root) / Path(self.subdir)), **kwargs)

        train, test = split(data, train_len, test_len)
        train = TransformDataset(train, None, self.train_transform)
        test = TransformDataset(test, None, self.test_transform)
        self.shape = train[0][0].shape
        return train, test


class Builtin(DataPack):
    def __init__(self, torchv_class, train_augment, test_augment, train_baseline=None, test_baseline=None,
                 class_list=None, class_n=None):
        super().__init__()
        self.train_augment = train_augment
        self.test_augment = test_augment
        self.train_baseline = train_baseline
        self.test_baseline = test_baseline

        self.torchv_class = torchv_class
        self.add_classes(class_list, class_n)

    def make(self, train_len, test_len, data_root='data', **kwargs):
        train = self.torchv_class(data_root, train=True, transform=None, download=True)
        train = TransformDataset(train, baseline_tranform=self.train_baseline, augment_transform=self.train_augment)
        test = self.torchv_class(data_root, train=False, transform=None, download=True)
        test = TransformDataset(test, baseline_tranform=self.test_baseline, augment_transform=self.test_augment)

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
    'mnist': Builtin(tv.datasets.MNIST, datasets.mnist.train_transform, datasets.mnist.test_transform, class_n=10),
    'mnist_no_trans': Builtin(tv.datasets.MNIST, datasets.mnist.test_transform, datasets.mnist.test_transform,
                              class_n=10)
}


def register(key, datapack):
    datasets[key] = datapack


def get(key):
    return datasets[key]


def list():
    for key in datasets:
        print(key)
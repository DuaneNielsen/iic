import train_classifier
import config


def test_cifar10():
    args = config.config(['--config', '../configs/classify/cifar10_vgg16.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    train_classifier.main(args)


def test_mnist():
    args = config.config(['--config', '../configs/classify/mnist_vgg16.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    train_classifier.main(args)


def test_mnist_fc():
    args = config.config(['--config', '../configs/classify/mnist_fc.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    train_classifier.main(args)


def test_cifar10_resnet():
    args = config.config(['--config', '../configs/classify/cifar10_resnet.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    train_classifier.main(args)
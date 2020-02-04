import train_classifier
import config


def test_cifar10():
    args = config.config(['--config', '../configs/classify/cifar10_vgg16.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256',
                          '--seed', '0'
                          ])
    ave_precision, best_precision = train_classifier.main(args)
    assert ave_precision > 0.2
    assert best_precision > 0.2


def test_mnist():
    args = config.config(['--config', '../configs/classify/mnist_vgg16.yaml',
                          '--epochs', '5',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    ave_precision, best_precision = train_classifier.main(args)
    assert ave_precision > 0.5
    assert best_precision > 0.5


def test_mnist_fc():
    args = config.config(['--config', '../configs/classify/mnist_fc.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    ave_precision, best_precision = train_classifier.main(args)
    assert ave_precision > 0.2
    assert best_precision > 0.2


def test_cifar10_resnet():
    args = config.config(['--config', '../configs/classify/cifar10_resnet.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    ave_precision, best_precision = train_classifier.main(args)


def test_cifar10_resnet_fixup():
    args = config.config(['--config', '../configs/classify/cifar10_resnet_fixup.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256',
                          '--device', 'cuda:1'
                          ])
    ave_precision, best_precision = train_classifier.main(args)
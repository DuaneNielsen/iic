import train_classifier
import config

"""
These unit tests run quickly to check if the models show signs of life.
"""

"""
Things to check if the training is not stable...

1.  Initialization
2.  Normalization layers
2.  Learning rate too high - reduce it and increase epochs a bit

"""


def test_cifar10_vgg16():
    args = config.config(['--config', '../configs/classify/vgg16/cifar10.yaml',
                          '--optim_lr', '0.05',
                          '--epochs', '6',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256',
                          '--seed', '0',
                          '--run_id', '1'
                          ])
    ave_precision, best_precision, train_accuracy, test_accuracy = train_classifier.main(args)
    assert ave_precision > 0.2
    assert best_precision > 0.2


def test_mnist_vgg16():
    args = config.config(['--config', '../configs/classify/vgg16/mnist.yaml',
                          '--epochs', '5',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256',
                          '--seed', '0',
                          '--run_id', '2'
                          ])
    ave_precision, best_precision, train_accuracy, test_accuracy = train_classifier.main(args)
    assert ave_precision > 0.5
    assert best_precision > 0.5


def test_mnist_fc():
    args = config.config(['--config', '../configs/classify/fc/mnist.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256',
                          '--seed', '0',
                          '--run_id', '3'
                          ])
    ave_precision, best_precision, train_accuracy, test_accuracy = train_classifier.main(args)
    assert ave_precision > 0.2
    assert best_precision > 0.2


def test_cifar10_resnet():
    args = config.config(['--config', '../configs/classify/resnet/cifar10-batchnorm.yaml',
                          '--epochs', '80',
                          '--optim_lr', '0.01',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256',
                          '--seed', '0',
                          '--run_id', '4'
                          ])
    ave_precision, best_precision, train_accuracy, test_accuracy = train_classifier.main(args)

    """ WARNING this model does not run reliably due to the shortcut containing convnets"""
    assert best_precision > 0.13
    assert train_accuracy > 20.0


def test_cifar10_resnet_fixup():
    args = config.config(['--config', '../configs/classify/resnet/cifar10_fixup.yaml',
                          '--optim_lr', '0.05',
                          '--epochs', '80',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256',
                          '--seed', '0',
                          '--run_id', '5'
                          ])
    ave_precision, best_precision, train_accuracy, test_accuracy = train_classifier.main(args)
    assert ave_precision > 0.2
    assert best_precision > 0.2
    assert train_accuracy > 20.0


def test_cifar10_resnet_const_fixup():
    args = config.config(['--config', '../configs/classify/resnet/cifar10.yaml',
                          '--optim_lr', '0.01',
                          '--epochs', '80',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256',
                          '--seed', '0',
                          '--run_id', '6'
                          ])
    ave_precision, best_precision, train_accuracy, test_accuracy = train_classifier.main(args)
    assert ave_precision > 0.2
    assert best_precision > 0.2
    assert train_accuracy > 20.0
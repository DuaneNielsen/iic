import train_autoencoder
import config


def test_celeba():
    args = config.config(['--config', '../configs/autoencoder/celeba.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    train_autoencoder.main(args)


def test_cifar10():
    args = config.config(['--config', '../configs/autoencoder/cifar10.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    train_autoencoder.main(args)


def test_mnistfc():
    args = config.config(['--config', '../configs/autoencoder/mnist_fc.yaml',
                          '--epochs', '3',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    train_autoencoder.main(args)

import train_autoencoder
import config


def test_autoencode_vgg16_celeba():

    args = config.config(['--config', '../configs/autoencoder/vgg16/celeba.yaml',
                          '--epochs', '5',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    best_loss = train_autoencoder.main(args)
    assert best_loss < 0.04


def test_autoencode_vgg16_cifar10():

    args = config.config(['--config', '../configs/autoencoder/vgg16/cifar10.yaml',
                          '--epochs', '16',
                          '--dataroot', '../data',
                          '--dataset_test_len', '256',
                          '--dataset_train_len', '256'
                          ])
    best_loss = train_autoencoder.main(args)
    assert best_loss < 1.00


def test_autoencode_fc_mnist():

        args = config.config(['--config', '../configs/autoencoder/fc/mnist.yaml',
                              '--epochs', '80',
                              '--dataroot', '../data',
                              '--dataset_test_len', '256',
                              '--dataset_train_len', '256'
                              ])
        best_loss = train_autoencoder.main(args)
        assert best_loss < 1.3
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from datasets.package import register, Builtin
import scipy.signal as signal
import numpy as np
from PIL import Image

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

gray = T.Compose([
    T.Grayscale(),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

gray_jitter = T.Compose([
    T.ColorJitter(brightness=[0.5, 1], contrast=[0.5, 1], saturation=1, hue=0.5),
    #T.Grayscale(),
    T.ToTensor(),
])

sobel_x = np.c_[
    [-1,0,1],
    [-2,0,2],
    [-1,0,1]
]

sobel_y = np.c_[
    [1,2,1],
    [0,0,0],
    [-1,-2,-1]
]


def three_channel_grey(pic):
    image = np.asarray(pic)
    image = np.expand_dims(image, axis=2)
    image = np.repeat(image, 3, axis=2)
    return Image.fromarray(image)


def gradient(pic):
    image = np.asarray(pic)
    channels = []

    for i in range(3):
        sx = signal.convolve2d(image[:, :, i], sobel_x, boundary='symm', mode='same')
        sy = signal.convolve2d(image[:, :, i], sobel_y, boundary='symm', mode='same')
        channel = np.sqrt(sx ** 2 + sy ** 2)
        channels.append(channel)

    grad = np.stack(channels, axis=2)
    grad = (grad / np.max(grad) * 255).astype(np.uint8)
    return Image.fromarray(grad)


def gradient_grey(pic):
    image = np.asarray(pic)
    sx = signal.convolve2d(image[:, :], sobel_x, boundary='symm', mode='same')
    sy = signal.convolve2d(image[:, :], sobel_y, boundary='symm', mode='same')
    grad = np.sqrt(sx ** 2 + sy ** 2)

    grad = (grad / np.max(grad) * 255).astype(np.uint8)
    return Image.fromarray(grad)


class Gradient:
    def __call__(self, pic):
        return gradient(pic)


def clip_below(pic, value):
    image = np.asarray(pic).copy()
    image[image < value] = 0.0
    return Image.fromarray(image)


class ClipBelow:
    def __init__(self, value):
        self.value = value

    def __call__(self, pic):
        return clip_below(pic, self.value)


def connectedness(pic, mask_threshold, connect_threshold):
    image = np.asarray(pic).copy()
    connect_map = np.asarray(pic).copy()
    mask = image > mask_threshold
    connect_map[image >= mask_threshold] = 1.0
    neighbor_count = np.c_[
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]

    for i in range(40):
        connect_map = connect_map * mask
        connect_map = signal.convolve2d(connect_map[:, :], neighbor_count, boundary='symm', mode='same')

    connect_mask = connect_map > connect_threshold

    image = (image * connect_mask).astype(np.uint8)
    return Image.fromarray(image)


def convolve(pic, kernel):
    image = np.asarray(pic).copy()
    image = signal.convolve2d(image[:, :], kernel, boundary='symm', mode='same')
    image = (image / np.max(image) * 255).astype(np.uint8)
    return Image.fromarray(image)


def blur(pic, alpha=0.5):
    b = (1.0 - alpha) / 8
    kernel = np.c_[
        [b, b, b],
        [b, alpha, b],
        [b, b, b]
    ]
    return convolve(pic, kernel)


grad = T.Compose([
    Gradient(),
    T.Grayscale(),
    ClipBelow(100),
    T.ToTensor(),
])


class_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

register('cifar-10-no-aug', Builtin(CIFAR10,
                                    train_augment=T.ToTensor(),
                                    test_augment=T.ToTensor(),
                                    class_list=class_list)
         )

register('cifar-10', Builtin(CIFAR10,
                             train_augment=transform_train,
                             test_augment=transform_test,
                             class_list=class_list))

register('cifar-10-normal', Builtin(CIFAR10,
                                    train_augment=transform_train,
                                    test_augment=transform_test,
                                    class_list=class_list))

register('cifar-10-gray', Builtin(CIFAR10, train_augment=gray, test_augment=gray, class_list=class_list))

register('cifar-10-gray-jitter', Builtin(CIFAR10, train_augment=gray_jitter, test_augment=gray_jitter, class_list=class_list))

register('cifar-10-grad', Builtin(CIFAR10, train_baseline=grad, train_augment=grad, test_baseline=grad, test_augment=grad, class_list=class_list))
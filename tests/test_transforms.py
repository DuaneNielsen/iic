import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
import numpy as np
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.imagenet import ImageNet
from torchvision.transforms import Grayscale
import torchvision.transforms.functional as F
from datasets.cifar10 import *

to_grey = Grayscale()

def show(orig, middle, final=None):
    fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))
    ax_orig.imshow(orig)
    ax_orig.set_title('Original')
    ax_orig.set_axis_off()
    ax_mag.imshow(middle,  cmap='gray', vmin=0, vmax=255)
    ax_mag.set_title('Gradient magnitude')
    ax_mag.set_axis_off()
    if final:
        ax_ang.imshow(final, cmap='gray', vmin=0, vmax=255)
        ax_ang.set_title('Gradient magnitude')
        ax_ang.set_axis_off()
    fig.show()

def test_gradient():
    orig = misc.ascent()
    grad = gradient_grey(orig)
    grad_one = grad.copy()
    for i in range(5):
        grad = gradient_grey(grad)
    show(orig, grad_one, grad)


def test_gradient_color():
    to_grey = Grayscale()
    data = CIFAR10('/home/duane/PycharmProjects/iic/data', download=True)
    orig, label = data[5880]
    grad_color = gradient(orig)
    grey = to_grey(grad_color)

    grad = gradient_grey(grey)
    for i in range(5):
        grad = gradient_grey(grad)
    show(orig, grad_color, grad)


def test_blur():
    orig = misc.ascent()
    blurred = blur(orig)
    final = blur(blurred)
    for i in range(5):
        final=blur(final)
    show(orig, blurred, final)


def test_grad_blur():
    orig = misc.ascent()
    blurred = blur(gradient_grey(orig))
    final = blur(gradient_grey(blurred))
    for i in range(4):
        final=blur(gradient_grey(final))
    final = three_channel_grey(final)
    final = F.adjust_brightness(final, 2.0)
    final = to_grey(final)
    show(orig, blurred, final)


def test_grad_blur_color():
    data = CIFAR10('/home/duane/PycharmProjects/iic/data', download=True)
    orig, label = data[5810]
    orig = to_grey(orig)

    blurred = blur(gradient_grey(orig))
    final = blur(gradient_grey(blurred))
    for i in range(1):
        final=blur(gradient_grey(final))
    final = three_channel_grey(final)
    final = F.adjust_brightness(final, 2.0)
    final = to_grey(final)
    show(orig, blurred, final)


def test_three_channel_grey():
    to_grey = Grayscale()
    data = CIFAR10('/home/duane/PycharmProjects/iic/data')
    orig, label = data[100]

    grey = to_grey(orig)

    color = three_channel_grey(grey)

    show(orig, grey, color)


def test_connectedness():
    orig = misc.ascent()
    grad = gradient_grey(orig)
    final = connectedness(grad, 20.0, 100)
    show(orig, grad, final)


def test_connectedness_color():
    to_grey = Grayscale()
    data = CIFAR10('/home/duane/PycharmProjects/iic/data')
    orig, label = data[100]

    grad = gradient(orig)
    grey = to_grey(grad)
    image_np = np.asarray(grey)
    connected = connectedness(grey, 20.0, 0.5)
    image_np_con = np.asarray(connected)
    show(orig, grey, connected)

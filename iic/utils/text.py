import numpy as np
import pygame
import torch


def text_patch(text, patch_size, center=True, font=None, fontsize=12, forecolor = (255, 255, 255, 255), backcolor = (0, 0, 0, 255), return_numpy=False):
    """

    :param text: the text to put on the patch
    :param patch_size: the shape of the patch to create in (C, H, W) or (H, W)
    :param offset: offset to help center the image
    :param font: pygame font string
    :param fontsize:
    :param forecolor: tuple (R, G, B, A)
    :param backcolor: tuple (R, G, B, A)
    :param return_numpy: return a numpy array (C, H, W)
    :return:
    """
    font = pygame.font.Font(font, fontsize)
    textSurface = font.render(text, True, forecolor, backcolor)

    if len(patch_size) == 2:
        array = pygame.surfarray.array2d(textSurface).swapaxes(0, 1)
        h_axis = 0
    elif patch_size[0] == 1:
        array = pygame.surfarray.array2d(textSurface).swapaxes(0, 1)
        array = np.expand_dims(array, axis=0)
        h_axis = 1
    else:
        array = pygame.surfarray.array3d(textSurface).swapaxes(0, 2)
        h_axis = 1

    patch = np.zeros(patch_size, dtype=array.dtype)

    mins = [slice(0, min(a, b)) for a, b in zip(array.shape, patch.shape)]

    if len(mins) == 2:
        patch[mins[0], mins[1]] = array[mins[0], mins[1]]
    else:
        patch[mins[0], mins[1], mins[2]] = array[mins[0], mins[1], mins[2]]

    if center:
        patch = np.roll(patch, (patch.shape[h_axis] - array.shape[h_axis]) // 2, axis=h_axis)

    if return_numpy:
        return patch
    else:
        return torch.from_numpy(patch) / 255.0


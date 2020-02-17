import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

def to_numpyRGB(image, invert_color=False):
    """
    Universal method to detect and convert an image to numpy RGB format
    :params image: the output image
    :params invert_color: perform RGB -> BGR convert
    :return: the output image
    """

    if type(image) == Image.Image:
        img = image.convert("RGB")
        img = np.array(img)
        return img

    if type(image) == torch.Tensor:
        image = image.cpu().detach().numpy()
    # remove batch dimension
    if len(image.shape) == 4:
        image = image[0]
    smallest_index = None
    if len(image.shape) == 3:
        smallest = min(image.shape[0], image.shape[1], image.shape[2])
        smallest_index = image.shape.index(smallest)
    elif len(image.shape) == 2:
        smallest = 0
    else:
        raise Exception(f'too many dimensions, I got {len(image.shape)} dimensions, give me less dimensions')
    if smallest == 3:
        if smallest_index == 2:
            pass
        elif smallest_index == 0:
            image = np.transpose(image, [1, 2, 0])
        elif smallest_index == 1:
            # unlikely
            raise Exception(f'Is this a color image?')
        if invert_color:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif smallest == 1:
        image = np.squeeze(image)
    elif smallest == 0:
        # greyscale
        pass
    elif smallest == 4:
        # I guess its probably the 32-bit RGBA format
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        raise Exception(f'dont know how to display color of dimension {smallest}')
    return image


class UniImageViewer:
    def __init__(self, title='title', screen_resolution=(640, 480), format=None, channels=None, invert_color=True):
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution
        self.format = format
        self.channels = channels
        self.invert_color = invert_color

    def render(self, image, block=False):

        image = to_numpyRGB(image, self.invert_color)

        image = cv2.resize(image, self.screen_resolution)

        # Display the resulting frame
        cv2.imshow(self.title, image)
        if block:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)

    def view_input(self, model, input, output):
        image = input[0] if isinstance(input, tuple) else input
        self.render(image)

    def view_output(self, model, input, output):
        image = output[0] if isinstance(output, tuple) else output
        self.render(image)

    def update(self, image):
        self.render(image)


def make_grid(tensor, rows, columns):
    """
    Makes a grid from a B, C, H, W tensor
    Far superior to the torchvision version, coz it's vectorized

    If B > rows * columns, it will truncate for you.
    truncate using an index tensor before you pass in if you want specific images
    in specific slots

    :param tensor:
    :param rows:
    :param columns:
    :return:
    """

    b, c, h, w = tensor.shape
    b = min(rows * columns, b)
    grid = torch.ones(b, c, h, w, device=tensor.device, dtype=tensor.dtype)
    index = torch.arange(b, device=tensor.device)
    grid[index] = tensor[index]
    grid = torch.cat(grid.unbind(0), dim=1).unsqueeze(0)
    grid = F.unfold(grid, kernel_size=(h, w), stride=(h, w))
    grid = F.fold(grid, output_size=(rows * h, columns * w), kernel_size=(h, w), stride=(h, w))
    return grid

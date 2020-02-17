import datasets.package as package
from torchvision.transforms import *

package.register('pong-v1',
                 package.ImageDataPack('pong-v1', 'patches/pong', ToTensor(), ToTensor(),
                                       class_n=6))

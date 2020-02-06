# Deep Learning Template Project

This is my template project for deep learning.

Features

* Explicit Pytorch style main loop (if you know Pytorch you can read the main loop)
* Convention over configuration management from yaml files and command line
* Simple checkpoint and restore system
* Simple dataset management
* Layer builder for building the boring bits
* Tensorboard logging
* Real time visualization of layers and outputs during training

### Basic command usage
```commandline
train.py --config config/cifar10.yaml --display 10
```
Train an autoencoder on cifar10, displaying images every 10 batches

```commandline
train.py --config config/cifar10.yaml --display 10 --batchsize 64  --epochs 200 
```
Train an autoencoder on cifar10, with batch size 64 and for 200 passes through the training set

### Configuration

Configuration flags can be specified in argparse parameters, or in yaml files, or in both.

--config parameter is used to specify a yaml file to load parameters from.  The yaml file contents will be added to the 
argparse namespace object.

Precedence is
* Arguments from command line
* Arguments from the config file
* Default value if specified in config.py

Yaml files can contain nested name-value pairs and they will be flattened

```yaml
dataset:
  name: celeba
  train_len: 10000
  test_len: 1000
```

will be flattened to argparse arguments

```
--dataset_name celeba
--dataset_train_len 10000
--dataset_test_len: 1000
```

### Data package

A data package is an object that contains everything required to load the data for training.

```python
import datasets.package as package

datapack = package.datasets['celeba']

train, test = datapack.make(train_len=10000, test_len=400, data_root='data')

``` 

get a training set of length 1000 and a test set of length 400

### Example config

for training VGG16 on CIFAR 10 with a custom SGD schedule  

```yaml
batchsize: 128
epochs: 350

dataset:
  name: cifar-10-normed

model:
  name: VGG16
  type: conv
  encoder: [3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

optim:
  class: SGD
  lr: 0.1
  momentum: 0.9
  weight_decay: 5e-4

scheduler:
  class: MultiStepLR
  milestones: [150, 250]
  gamma: 0.1
```

see more example configs in the configs directory of the project

### Configuring the optimizers

```yaml
optim:
  class: SGD
  lr: 0.1
  momentum: 0.9
  weight_decay: 5e-4

scheduler:
  class: MultiStepLR
  milestones: [150, 250]
  gamma: 0.1
```

and in the code

```python
    import config
    import torch.nn as nn

    args = config.config()
    model = nn.Linear(10, 2)
    optim, scheduler = config.get_optim(args, model.parameters())
```

### Layer builder

If you get bored of typing the same NN blocks over and over, you can instead use the layer builder.

It works similar to the Pytorch built-in layer builder, it can build

fully connected: type = 'fc' vgg: type = 'vgg' or resnet: type = 'resnet'

for example, to build vgg blocks...

```python
from models.mnn import make_layers
from models.layerbuilder import LayerMetaData

meta = LayerMetaData(input_shape=(32, 32))

encoder_core, meta = make_layers(['C:3', 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], type='vgg', meta=meta)

decoder_core, meta = make_layers([512, 512, 'U', 256, 256, 'U', 256, 256, 'U', 128, 'U', 64, 'U', 'C:3'], type='vgg', meta=meta)
```

M -> Max Pooling
U -> Linear Upsample
C:3 -> Conv layer with 3 channels

### Duplicating this project

```commandline
git clone --bare https://github.com/DuaneNielsen/deep_learning_base.git
cd deep_learning_base.git/
git push --mirror https://github.com/DuaneNielsen/<NEW REPO NAME>.git
```
import torch
import torch.nn as nn
from models.mnn import initialize_weights


class OutputBlock(nn.Module):
    def __init__(self, num_classes, in_channels=512):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, num_classes),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class Vectorizer(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()

        in_dim = 1
        for i in input_shape:
            in_dim = in_dim * i
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class Classifier(nn.Module):
    def __init__(self, encoder, encoder_output_shape, num_classes, init_weights=True):
        super().__init__()
        self.encoder = encoder
        self.output_block = Vectorizer(num_classes, encoder_output_shape)

        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        h = self.encoder(x)
        return self.output_block(h)
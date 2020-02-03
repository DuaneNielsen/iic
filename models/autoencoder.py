import torch
import torch.nn as nn
from .mnn import initialize_weights


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder,
                 init_weights=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if init_weights:
            initialize_weights(self)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return z, x
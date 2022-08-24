import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gram_discriminator import *


class Discriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.step = 0


class GramDiscriminator(Discriminator):
    def __init__(self, img_size, img_channels=3): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.img_size = img_size
        self.img_size_log2 = int(np.log2(img_size))
        self.layers = nn.ModuleList([])
        for i in range(13 - self.img_size_log2, 12):
            self.layers.append(ResidualCoordConvBlock(int(min(400, 2**i)), int(min(400, 2**(i+1))), downsample=True))

        self.fromRGB = AdapterBlock(img_channels, int(min(400, 2**(13 - self.img_size_log2))))

        self.final_layer = nn.Conv2d(400, 1 + 2, 2)

    def forward(self, input):
        x = self.fromRGB(input)
        for layer in self.layers:
            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)

        prediction = x[..., 0:1]
        position = x[..., 1:]

        return prediction, position

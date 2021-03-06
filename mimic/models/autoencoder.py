from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch._C import device
from typing import Dict

from mimic.primitives import AbstractEncoder
from .common import _Model, NullConfig
from .common import LossDict

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class DeepEncoder(AbstractEncoder):
    encoder: nn.Module
    def __init__(self, encoder, size_input, n_output):
        super().__init__(size_input, n_output)
        self.encoder = encoder
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

class AbstractEncoderDecoder(_Model[NullConfig], ABC):
    encoder : nn.Module
    decoder : nn.Module
    n_bottleneck : int

    def __init__(self, device: device, n_bottleneck: int, **kwargs):
        _Model.__init__(self, device, NullConfig())
        self.n_bottleneck = n_bottleneck
        self._create_layers(**kwargs)

    def forward(self, inp : torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inp))

    def get_encoder(self) -> DeepEncoder:
        # TODO size is currently -1 tuple
        return DeepEncoder(self.encoder, (-1,), self.n_bottleneck)

    def compat_modelconfig(cls): return NullConfig

class ImageAutoEncoder(AbstractEncoderDecoder):

    def loss(self, sample : torch.Tensor) -> LossDict:
        f_loss = nn.MSELoss()
        reconstructed = self.forward(sample)
        loss_value = f_loss(sample, reconstructed)
        return LossDict({'reconstruction': loss_value})

    def _create_layers(self, **kwargs):
        channel, n_pixel, m_pixel = kwargs['image_shape']
        assert n_pixel == m_pixel
        assert n_pixel in [28, 112, 224]

        # TODO do it programatically
        if n_pixel==224:
            self.encoder = nn.Sequential(
                nn.Conv2d(channel, 8, 3, padding=1, stride=(2, 2)), # 112x112
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)), # 56x56
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)), # 28x28
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)), # 14x14
                nn.ReLU(inplace=True), # 64x4x4
                nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)), # 7x7
                nn.ReLU(inplace=True), # 64x4x4
                nn.Conv2d(128, 256, 3, padding=1, stride=(2, 2)), # 4x4
                nn.ReLU(inplace=True), # 64x4x4
                nn.Flatten(),
                nn.Linear(256 * 16, 512),
                nn.Linear(512, self.n_bottleneck)
                )
            self.decoder = nn.Sequential(
                nn.Linear(self.n_bottleneck, 512),
                nn.Linear(512, 256 * 16),
                nn.ReLU(inplace=True),
                Reshape(-1, 256, 4, 4),
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(8, channel, 4, stride=2, padding=1),
                nn.Sigmoid(),
                )
        elif n_pixel==112:
            self.encoder = nn.Sequential(
                nn.Conv2d(channel, 8, 3, padding=1, stride=(2, 2)), # 56x56
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)), # 28x28
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)), # 14x14
                nn.ReLU(inplace=True), # 64x4x4
                nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)), # 7x7
                nn.ReLU(inplace=True), # 64x4x4
                nn.Conv2d(64, 128, 3, padding=1, stride=(2, 2)), # 4x4
                nn.ReLU(inplace=True), # 64x4x4
                nn.Flatten(),
                nn.Linear(128 * 16, 512),
                nn.Linear(512, self.n_bottleneck)
                )
            self.decoder = nn.Sequential(
                nn.Linear(self.n_bottleneck, 512),
                nn.Linear(512, 128 * 16),
                nn.ReLU(inplace=True),
                Reshape(-1, 128, 4, 4),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(8, channel, 4, stride=2, padding=1),
                nn.Sigmoid(),
                )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(channel, 8, 3, padding=1, stride=(2, 2)), # 14x14
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)), # 7x7
                nn.ReLU(inplace=True), # 64x4x4
                nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)), # 4x4
                nn.ReLU(inplace=True), # 64x4x4
                nn.Flatten(),
                nn.Linear(32 * 16, 8 * 16),
                nn.Linear(8 * 16, self.n_bottleneck)
                )
            self.decoder = nn.Sequential(
                nn.Linear(self.n_bottleneck, 8 * 16),
                nn.Linear(8 * 16, 32 * 16),
                nn.ReLU(inplace=True),
                Reshape(-1, 32, 4, 4),
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(8, channel, 4, stride=2, padding=1),
                nn.Sigmoid(),
                )

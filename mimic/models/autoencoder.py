from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class AbstractEncoderDecoder(ABC, nn.Module):
    encoder : nn.Module
    decoder : nn.Module
    n_bottleneck : int
    device : torch.device

    def __init__(self, n_bottleneck, device, **kwargs):
        nn.Module.__init__(self)
        self.n_bottleneck = n_bottleneck
        self.device = device
        self._create_layers(**kwargs)

    @abstractmethod
    def _create_layers(self, **kwargs) -> None: ...

    @abstractmethod
    def _loss(self, sample : Any) -> torch.Tensor: ...

    def loss(self, sample : Any) -> torch.Tensor:
        sample = sample.to(self.device)
        loss = self._loss(sample)
        return loss

    def forward(self, inp : torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inp))

class AbstractAutoEncoder(AbstractEncoderDecoder):
    def _loss(self, sample : torch.Tensor) -> torch.Tensor:
        f_loss = nn.MSELoss()
        reconstructed = self.forward(sample)
        return f_loss(reconstructed, sample)

class ImageAutoEncoder(AbstractAutoEncoder):
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

import pytest
import torch

from mimic.models import ImageAutoEncoder

def test_image_auto_encoder():
    n_batch = 100
    n_channel = 3
    for n_pixel in [28, 112, 224]:
        sample = torch.randn(n_batch, n_channel, n_pixel, n_pixel)
        ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))

        reconstructed = ae(sample)
        assert reconstructed.shape == sample.shape
        loss = ae.loss(sample)
        assert len(list(loss.values())) == 1
        assert float(loss['reconstruction'].item()) > 0.0 # check if positive scalar

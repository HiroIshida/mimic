import pytest
import torch

from mimic.models import ImageAutoEncoder
from mimic.dataset import AutoRegressiveDataset
from mimic.models import LSTM
from test_datatypes import image_datachunk_with_encoder
from test_datatypes import cmd_datachunk

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

def test_lstm_with_image(image_datachunk_with_encoder): 
    dataset = AutoRegressiveDataset.from_chunk(image_datachunk_with_encoder)
    n_seq, n_state = dataset.data[0].shape 
    model = LSTM(torch.device('cpu'), n_state)
    sample = dataset.data[0].unsqueeze(0)
    loss = model.loss(sample)
    assert len(list(loss.values())) == 1
    assert float(loss['prediction'].item()) > 0.0 # check if positive scalar 

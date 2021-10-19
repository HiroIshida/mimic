import pytest
import torch

from mimic.models import ImageAutoEncoder
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset
from mimic.models import LSTM
from mimic.models import DenseProp
from mimic.models import BiasedDenseProp
from test_datatypes import cmd_datachunk
from test_datatypes import image_datachunk_with_encoder
from test_datatypes import image_command_datachunk_with_encoder

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

        encoder = ae.get_encoder()
        encoder(sample)

def test_lstm_with_image(image_datachunk_with_encoder): 
    dataset = AutoRegressiveDataset.from_chunk(image_datachunk_with_encoder)
    n_seq, n_state = dataset.data[0].shape 
    model = LSTM(torch.device('cpu'), n_state)
    sample = dataset.data[0].unsqueeze(0)
    loss = model.loss(sample)
    assert len(list(loss.values())) == 1
    assert float(loss['prediction'].item()) > 0.0 # check if positive scalar 

def test_densedrop_pipeline(image_command_datachunk_with_encoder):
    chunk = image_command_datachunk_with_encoder
    dataset = FirstOrderARDataset.from_chunk(chunk)
    n_state = 16 + 7
    model = DenseProp(torch.device('cpu'), n_state)
    pre, post = dataset[0]
    sample = (pre.unsqueeze(0), post.unsqueeze(0))
    loss_dict = model.loss(sample)

def test_biaseddensedrop_pipeline(image_command_datachunk_with_encoder):
    chunk = image_command_datachunk_with_encoder
    dataset = BiasedFirstOrderARDataset.from_chunk(chunk)
    n_state, n_bias = 7, 16
    model = BiasedDenseProp(torch.device('cpu'), n_state, n_bias)
    pre, post, bias = dataset[0]
    sample = (pre.unsqueeze(0), post.unsqueeze(0), bias.unsqueeze(0))
    loss_dict = model.loss(sample)

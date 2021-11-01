import pytest
import torch

from mimic.models import ImageAutoEncoder
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset
from mimic.models import LSTM, LSTMConfig
from mimic.models import BiasedLSTM
from mimic.models import DenseProp, DenseConfig
from mimic.models import BiasedDenseProp
from mimic.models.denseprop import create_linear_layers
from mimic.models.denseprop import KinemaNet
from test_datatypes import cmd_datachunk
from test_datatypes import image_datachunk_with_encoder
from test_datatypes import image_command_datachunk_with_encoder
from test_dataset import kinematics_dataset

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
    model = LSTM(torch.device('cpu'), n_state, LSTMConfig())
    sample_ = dataset[0]
    assert isinstance(sample_, tuple)
    sample = (sample_[0].unsqueeze(0), sample_[1].unsqueeze(0))
    loss = model.loss(sample)
    assert len(list(loss.values())) == 1
    assert float(loss['prediction'].item()) > 0.0 # check if positive scalar 

    loss_sliced = model.loss(sample, slice(5, None))

def test_biased_lstm_pipeline(image_command_datachunk_with_encoder):
    dataset = BiasedAutoRegressiveDataset.from_chunk(image_command_datachunk_with_encoder)
    model = BiasedLSTM(torch.device('cpu'), dataset.n_state, dataset.n_bias, LSTMConfig())
    sample_ = dataset[0]
    assert isinstance(sample_, tuple)
    sample = (sample_[0].unsqueeze(0), sample_[1].unsqueeze(0))
    loss = model.loss(sample)
    assert len(list(loss.values())) == 1
    assert float(loss['prediction'].item()) > 0.0 # check if positive scalar 

def test_create_linear_layers():
    for activation in ['relu', 'tanh', 'sigmoid']:
        layers = create_linear_layers(10, 10, 20, 2, activation)
        assert len(layers) == 7
    layers = create_linear_layers(10, 10, 20, 2, None)
    assert len(layers) == 4

def test_denseprop_pipeline(image_command_datachunk_with_encoder):
    chunk = image_command_datachunk_with_encoder
    dataset = AutoRegressiveDataset.from_chunk(chunk)
    n_state = 16 + 7 + 1
    model = DenseProp(torch.device('cpu'), n_state, DenseConfig(activation='leru'))
    pre, post = dataset[0]
    sample = (pre.unsqueeze(0), post.unsqueeze(0))
    loss = model.loss(sample)
    assert len(list(loss.values())) == 1
    assert float(loss['prediction'].item()) > 0.0 # check if positive scalar 

def test_biaseddenseprop_pipeline(image_command_datachunk_with_encoder):
    chunk = image_command_datachunk_with_encoder
    dataset = BiasedAutoRegressiveDataset.from_chunk(chunk)
    sample_ = dataset[0]
    sample = (sample_[0].unsqueeze(0), sample_[1].unsqueeze(0))

    n_state, n_bias = dataset.n_state, 16
    model = BiasedDenseProp(torch.device('cpu'), n_state, n_bias, DenseConfig())
    pred = model.forward(sample[0])
    assert pred.shape == sample[1].shape

    loss = model.loss(sample)
    assert len(list(loss.values())) == 1
    assert float(loss['prediction'].item()) > 0.0 # check if positive scalar 

def test_kinemanet_pipeline(kinematics_dataset):
    dataset = kinematics_dataset
    model = KinemaNet(torch.device('cpu'), dataset.meta_data, DenseConfig())
    pre, post = dataset[0]
    sample = (pre.unsqueeze(0), post.unsqueeze(0))
    loss = model.loss(sample)

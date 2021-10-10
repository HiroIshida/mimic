import pytest
import torch

from mimic.dataset import AutoRegressiveDataset
from mimic.models import LSTM

from test_datatypes import image_datachunk_with_encoder
from test_datatypes import cmd_datachunk

def test_lstm_with_image(image_datachunk_with_encoder): 
    dataset = AutoRegressiveDataset.from_chunk(image_datachunk_with_encoder)
    n_seq, n_state = dataset.data[0].shape 
    model = LSTM(torch.device('cpu'), n_state)
    sample = dataset.data[0].unsqueeze(0)
    loss = model.loss(sample)
    assert len(list(loss.values())) == 1
    assert float(loss['prediction'].item()) > 0.0 # check if positive scalar 

    out = model(torch.randn(1, 10, 16))
    assert list(out.shape) == [1, 10, 16] # flag info must be stripped
    with pytest.raises(AssertionError):
        model(torch.randn(2, 10, 16))

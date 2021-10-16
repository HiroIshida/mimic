import numpy as np
import torch
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.predictor import ImageLSTMPredictor
from mimic.predictor import LSTMPredictor
from mimic.datatype import CommandDataChunk
from mimic.dataset import AutoRegressiveDataset

def test_predictor_core():
    chunk = CommandDataChunk()
    seq = np.random.randn(50, 7)
    for i in range(10):
        chunk.push_epoch(seq)
    dataset = AutoRegressiveDataset.from_chunk(chunk)
    """
    seq = dataset[0][:29, :7]

    lstm = LSTM(torch.device('cpu'), 7 + 1)
    predictor = LSTMPredictor(lstm)
    for cmd in seq:
        predictor.feed(cmd.detach().numpy())
    assert torch.all(torch.stack(predictor.states) == seq)

    cmd_pred = predictor.predict(n_horizon=1, with_feeds=False)

    out = lstm(torch.unsqueeze(dataset[0][:30, :], dim=0))
    cmd_pred_direct = out[0][-1, :-1].detach().numpy()
    assert np.all(cmd_pred == cmd_pred_direct)
    """

def test_ImageLSTMPredictor():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    lstm = LSTM(torch.device('cpu'), 17)
    predictor = ImageLSTMPredictor(lstm, ae)

    for _ in range(10):
        img = np.zeros((n_pixel, n_pixel, n_channel))
        predictor.feed(img)
    assert len(predictor.states) == 10
    assert list(predictor.states[0].shape) == [16 + 1] # flag must be attached

    imgs = predictor.predict(5)
    assert len(imgs) == 5
    assert imgs[0].shape == (n_pixel, n_pixel, n_channel)

    imgs_with_feeds = predictor.predict(5, with_feeds=True)
    assert len(imgs_with_feeds) == (5 + 10)

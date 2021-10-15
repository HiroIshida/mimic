import numpy as np
import torch
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.predictor import ImageLSTMPredictor

def test_ImageLSTMPredictor():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    lstm = LSTM(torch.device('cpu'), 17)
    predictor = ImageLSTMPredictor(ae, lstm)

    for _ in range(10):
        img = np.zeros((n_pixel, n_pixel, n_channel))
        predictor.feed(img)
    assert len(predictor.states) == 10
    assert list(predictor.states[0].shape) == [16]

    imgs = predictor.predict(5)
    assert len(imgs) == 5
    assert imgs[0].shape == (n_pixel, n_pixel, n_channel)
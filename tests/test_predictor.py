import numpy as np
import torch
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.models import DenseProp
from mimic.models import BiasedDenseProp
from mimic.predictor import SimplePredictor
from mimic.predictor import ImagePredictor
from mimic.predictor import ImageCommandPredictor
from mimic.predictor import FFImageCommandPredictor
from mimic.predictor import get_model_specific_state_slice
from mimic.datatype import CommandDataChunk
from mimic.dataset import AutoRegressiveDataset

def test_predictor_core():
    chunk = CommandDataChunk()
    seq = np.random.randn(50, 7)
    for i in range(10):
        chunk.push_epoch(seq)
    dataset = AutoRegressiveDataset.from_chunk(chunk)
    seq = dataset[0][:29, :7]

    lstm = LSTM(torch.device('cpu'), 7 + 1)
    predictor = SimplePredictor(lstm)
    for cmd in seq:
        predictor.feed(cmd.detach().numpy())
    assert torch.all(torch.stack(predictor.states) == dataset[0][:29, :])

    cmd_pred = predictor.predict(n_horizon=1, with_feeds=False)

    out = lstm(torch.unsqueeze(dataset[0][:29, :], dim=0))
    cmd_pred_direct = out[0][-1, :-1].detach().numpy()
    assert np.all(cmd_pred == cmd_pred_direct)

def test_ImagePredictor():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    lstm = LSTM(torch.device('cpu'), 17)
    denseprop = DenseProp(torch.device('cpu'), 16)

    for propagator in [lstm, denseprop]:
        print('testing : {}'.format(propagator.__class__.__name__))
        predictor = ImagePredictor(propagator, ae)

        for _ in range(10):
            img = np.zeros((n_pixel, n_pixel, n_channel))
            predictor.feed(img)
        assert len(predictor.states) == 10
        if isinstance(propagator, LSTM):
            assert list(predictor.states[0].shape) == [16 + 1] # flag must be attached
        else:
            assert list(predictor.states[0].shape) == [16] # flag must be attached

        imgs = predictor.predict(5)
        assert len(imgs) == 5
        assert imgs[0].shape == (n_pixel, n_pixel, n_channel)

        imgs_with_feeds = predictor.predict(5, with_feeds=True)
        assert len(imgs_with_feeds) == (5 + 10)

def test_ImageCommandPredictor():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    lstm = LSTM(torch.device('cpu'), 16 + 7 + 1)
    denseprop = DenseProp(torch.device('cpu'), 16 + 7)

    for propagator in [lstm, denseprop]:
        print('testing : {}'.format(propagator.__class__.__name__))
        predictor = ImageCommandPredictor(propagator, ae)

        for _ in range(10):
            img = np.zeros((n_pixel, n_pixel, n_channel))
            cmd = np.zeros(7)
            predictor.feed((img, cmd))
        if isinstance(propagator, LSTM):
            assert list(predictor.states[0].shape) == [16 + 7 + 1]
        else:
            assert list(predictor.states[0].shape) == [16 + 7]

        imgs, cmds = zip(*predictor.predict(5))
        assert imgs[0].shape == (n_pixel, n_pixel, n_channel)
        assert cmds[0].shape == (7,)

def test_FFImageCommandPredictor():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    prop = BiasedDenseProp(torch.device('cpu'), 7, 16)
    predictor = FFImageCommandPredictor(prop, ae)

    assert predictor.img_torch_one_shot is None
    for _ in range(10):
        img = np.zeros((n_pixel, n_pixel, n_channel))
        cmd = np.zeros(7)
        predictor.feed((img, cmd))
    assert predictor.img_torch_one_shot is not None
    assert list(predictor.img_torch_one_shot.shape) == [16]
    assert list(predictor.states[0].shape) == [7]
    imgs, cmds = zip(*predictor.predict(5))

def test_evaluate_command_prop():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    biased_prop = BiasedDenseProp(torch.device('cpu'), 7, 16)
    lstm = LSTM(torch.device('cpu'), 16 + 7 + 1)

    slice1 = get_model_specific_state_slice(ae, biased_prop)
    assert slice1.start == 16
    assert slice1.stop == None
    assert slice1.step == None

    slice2 = get_model_specific_state_slice(ae, lstm)
    assert slice2.start == 16
    assert slice2.stop == -1
    assert slice2.step == None

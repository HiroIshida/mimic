import copy
import numpy as np
import torch
from mimic.models import ImageAutoEncoder
from mimic.models import LSTMBase, LSTMConfig, BiasedLSTMConfig
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import DenseBase, DenseConfig, BiasedDenseConfig
from mimic.models import DenseProp
from mimic.models import DeprecatedDenseProp, DeprecatedDenseConfig
from mimic.models import BiasedDenseProp
from mimic.models import AugedLSTM, AugedLSTMConfig
from mimic.robot import KukaSpec
from mimic.predictor import SimplePredictor
from mimic.predictor import evaluate_command_prediction_error_old
from mimic.predictor import evaluate_command_prediction_error
from mimic.predictor import ImagePredictor
from mimic.predictor import ImageCommandPredictor
from mimic.predictor import FFImageCommandPredictor
from mimic.predictor import get_model_specific_state_slice
from mimic.datatype import CommandDataChunk, FeatureInfo, ImageCommandDataChunk
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset
from mimic.dataset import _continue_flag

from test_datatypes import image_command_datachunk_with_encoder

def assert_batch_seq_prediction_consistency(predictor_, input_init):
    # check if pred values from batch predict and sequencial predict are the same
    n_predict = 5

    # first compute batch predict values
    predictor = copy.deepcopy(predictor_)
    predictor.feed(input_init)
    preds_batch = predictor.predict(n_predict)

    # second compute batch predict values
    predictor = copy.deepcopy(predictor_)
    predictor.feed(input_init)
    preds_sequential = []
    for i in range(n_predict):
        pred_value = predictor.predict(1)[0]
        predictor.feed(pred_value)
        preds_sequential.append(pred_value)

    pred_seqs_batch = list(map(list, zip(*preds_batch)))
    pred_seqs_sequential = list(map(list, zip(*preds_sequential)))
    for i in range(len(pred_seqs_batch)):
        # for each component of predicted values (like img, cmd, ...)
        np.testing.assert_almost_equal(np.stack(pred_seqs_batch[i]), np.stack(pred_seqs_sequential[i]), decimal=1e-4)

def test_predictor_core():
    chunk = CommandDataChunk()
    seq = np.random.randn(50, 7)
    for i in range(10):
        chunk.push_epoch(seq)
    dataset = AutoRegressiveDataset.from_chunk(chunk)
    sample_input = dataset[0][0]
    seq = sample_input[:29, :7]

    config = LSTMConfig.from_finfo(chunk.get_feature_info())
    lstm = LSTM(torch.device('cpu'), config)
    predictor = SimplePredictor(lstm)
    for cmd in seq:
        predictor.feed(cmd.detach().numpy())
    seq_with_flag = torch.cat(
            (seq, torch.ones(29, 1) * _continue_flag), dim=1)
    assert torch.all(torch.stack(predictor.states) == seq_with_flag)

    cmd_pred = predictor.predict(n_horizon=1, with_feeds=False)

    out = lstm(torch.unsqueeze(seq_with_flag, dim=0))
    cmd_pred_direct = out[0][-1, :-1].detach().numpy()
    assert np.all(cmd_pred == cmd_pred_direct)

    assert_batch_seq_prediction_consistency(SimplePredictor(lstm), seq[0].detach().numpy())

def test_ImagePredictor():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    finfo = FeatureInfo(n_img_feature=16)
    lstm = LSTM(torch.device('cpu'), LSTMConfig.from_finfo(finfo))
    denseprop = DenseProp(torch.device('cpu'), DenseConfig.from_finfo(finfo))

    for propagator in [lstm, denseprop]:
        print('testing : {}'.format(propagator.__class__.__name__))
        predictor = ImagePredictor(propagator, ae)

        init_input = np.zeros((n_pixel, n_pixel, n_channel))
        assert_batch_seq_prediction_consistency(predictor, init_input)

        for _ in range(10):
            img = np.zeros((n_pixel, n_pixel, n_channel))
            predictor.feed(img)
        assert len(predictor.states) == 10
        if isinstance(propagator, (LSTMBase, DenseBase)):
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
    finfo = FeatureInfo(n_img_feature=16, n_cmd_feature=7)
    lstm = LSTM(torch.device('cpu'), LSTMConfig.from_finfo(finfo))
    denseprop = DenseProp(torch.device('cpu'), DenseConfig.from_finfo(finfo))
    depredense = DeprecatedDenseProp(torch.device('cpu'), DeprecatedDenseConfig.from_finfo(finfo))

    finfo = FeatureInfo(n_img_feature=16, n_cmd_feature=7, n_aug_feature=6)
    auged_lstm = AugedLSTM(torch.device('cpu'), AugedLSTMConfig.from_finfo(finfo, KukaSpec()))

    for propagator in [lstm, denseprop, depredense, auged_lstm]:
        print('testing : {}'.format(propagator.__class__.__name__))
        predictor = ImageCommandPredictor(propagator, ae)

        init_input = (np.zeros((n_pixel, n_pixel, n_channel)), np.zeros(7))
        assert_batch_seq_prediction_consistency(predictor, init_input)

        for _ in range(10):
            img = np.zeros((n_pixel, n_pixel, n_channel))
            cmd = np.zeros(7)
            predictor.feed((img, cmd))

        if isinstance(propagator, (AugedLSTM)):
            assert list(predictor.states[0].shape) == [16 + 7 + 6 + 1] # flag must be attached
        if isinstance(propagator, (LSTM, DenseProp)):
            assert list(predictor.states[0].shape) == [16 + 7 + 1] # flag must be attached
        if isinstance(propagator, (DeprecatedDenseProp)):
            assert list(predictor.states[0].shape) == [16 + 7] # flag must be attached

        imgs, cmds = zip(*predictor.predict(5))
        assert imgs[0].shape == (n_pixel, n_pixel, n_channel)
        assert cmds[0].shape == (7,)

def test_FFImageCommandPredictor():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    finfo = FeatureInfo(n_img_feature=16, n_cmd_feature=7)
    prop1 = BiasedLSTM(torch.device('cpu'), BiasedLSTMConfig.from_finfo(finfo))
    predictor1 = FFImageCommandPredictor(prop1, ae)
    prop2 = BiasedDenseProp(torch.device('cpu'), BiasedDenseConfig.from_finfo(finfo))
    predictor2 = FFImageCommandPredictor(prop2, ae)

    for predictor in [predictor1, predictor2]:
        print('testing : {}'.format(predictor.propagator.__class__.__name__))
        assert predictor.img_torch_one_shot is None
        for _ in range(10):
            img = np.zeros((n_pixel, n_pixel, n_channel))
            cmd = np.zeros(7)
            predictor.feed((img, cmd))

            assert predictor.img_torch_one_shot is not None
            assert list(predictor.img_torch_one_shot.shape) == [16]
            assert list(predictor.states[0].shape) == [7 + 16 + 1]

            imgs, cmds = zip(*predictor.predict(5))
            assert imgs[0] == None
            assert list(cmds[0].shape) == [7]

            imgs, cmds = zip(*predictor.predict(5, with_feeds=True))
            assert imgs[0] == None
            assert list(cmds[0].shape) == [7]

def test_evaluate_command_prop(image_command_datachunk_with_encoder):
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    biased_prop = BiasedDenseProp(torch.device('cpu'), BiasedDenseConfig(7 + 1, 16))
    dense_prop = DenseProp(torch.device('cpu'), DenseConfig(16 + 7 + 1))
    depre_prop = DeprecatedDenseProp(torch.device('cpu'), DeprecatedDenseConfig(16 + 7))
    lstm = LSTM(torch.device('cpu'), LSTMConfig(16 + 7 + 1))

    for model in [lstm, biased_prop]:
        slicer = get_model_specific_state_slice(ae, lstm)
        assert slicer.start == 16
        assert slicer.stop == -1
        assert slicer.step == None

    chunk = image_command_datachunk_with_encoder
    dataset = AutoRegressiveDataset.from_chunk(chunk)
    error = evaluate_command_prediction_error_old(ae, lstm, dataset)
    error2 = evaluate_command_prediction_error_old(ae, lstm, dataset, batch_size=2)
    assert abs(error2 - error) < 1e-3

    dataset = AutoRegressiveDataset.from_chunk(chunk)
    error = evaluate_command_prediction_error_old(ae, dense_prop, dataset)
    error2 = evaluate_command_prediction_error_old(ae, dense_prop, dataset, batch_size=2)
    assert abs(error2 - error) < 1e-3

    dataset = BiasedAutoRegressiveDataset.from_chunk(chunk)
    error = evaluate_command_prediction_error_old(ae, biased_prop, dataset)
    error2 = evaluate_command_prediction_error_old(ae, biased_prop, dataset, batch_size=2)
    assert abs(error2 - error) < 1e-3

    dataset = FirstOrderARDataset.from_chunk(chunk)
    error = evaluate_command_prediction_error_old(ae, depre_prop, dataset)
    error2 = evaluate_command_prediction_error_old(ae, depre_prop, dataset, batch_size=2)
    assert abs(error2 - error) < 1e-3

def test_evaluate_command_prop2(image_command_datachunk_with_encoder):
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))

    finfo = FeatureInfo(n_img_feature=16, n_cmd_feature=7)
    biased_prop = BiasedDenseProp(torch.device('cpu'), BiasedDenseConfig.from_finfo(finfo))
    dense_prop = DenseProp(torch.device('cpu'), DenseConfig.from_finfo(finfo))
    depre_prop = DeprecatedDenseProp(torch.device('cpu'), DeprecatedDenseConfig.from_finfo(finfo))
    lstm = LSTM(torch.device('cpu'), LSTMConfig.from_finfo(finfo))

    chunk: ImageCommandDataChunk = image_command_datachunk_with_encoder
    chunk_test, _ = chunk.split(1)

    prop_models = [biased_prop, dense_prop, depre_prop, lstm]
    for prop in prop_models:
        evaluate_command_prediction_error(ae, prop, chunk_test)


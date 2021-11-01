import pytest
import os
import shutil
import torch
from torch.utils.data import random_split
from mimic.file import get_project_dir
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM, LSTMConfig
from mimic.datatype import CommandDataChunk
from mimic.dataset import ReconstructionDataset
from mimic.dataset import AutoRegressiveDataset
from mimic.scripts.utils import split_with_ratio
from mimic.trainer import Config
from mimic.trainer import train
from mimic.trainer import TrainCache

from test_datatypes import image_datachunk
from test_datatypes import image_datachunk_with_encoder

def _train(project_name, model, dataset, model_type, config, postfix=""):
    n_total = len(dataset)
    train_set, val_set =  random_split(dataset, [n_total-2, 2])
    tcache = TrainCache[model_type](project_name, model.__class__, cache_postfix=postfix)

    assert not tcache.exists_cache()
    train(model, train_set, val_set, tcache=tcache, config=config)
    assert tcache.exists_cache()

    tcache = TrainCache.load(project_name, model.__class__, postfix)
    assert isinstance(tcache.best_model, model.__class__)

def test_train(image_datachunk, image_datachunk_with_encoder):
    config = Config(n_epoch=2)
    project_name = '__pytest'

    project_cache_path = get_project_dir(project_name)
    if os.path.exists(project_cache_path):
        shutil.rmtree(project_cache_path)

    dataset = ReconstructionDataset.from_chunk(image_datachunk)
    model = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(3, 28, 28))
    _train(project_name, model, dataset, ImageAutoEncoder, config)

    postfix = "_hogehoge"
    dataset2 = AutoRegressiveDataset.from_chunk(image_datachunk_with_encoder)
    n_seq, n_state = dataset2.data[0].shape 
    model2 = LSTM(torch.device('cpu'), n_state, LSTMConfig())
    _train(project_name, model2, dataset2, LSTM, config, postfix)

    # Another training session for LSTM using different LSTMConfig
    model3 = LSTM(torch.device('cpu'), n_state, LSTMConfig(100, 1))
    n_total = len(dataset2)
    train_set, val_set =  random_split(dataset2, [n_total-2, 2])
    tcache = TrainCache[LSTM](project_name, LSTM, cache_postfix=postfix)
    train(model3, train_set, val_set, tcache=tcache, config=config)

    with pytest.raises(AssertionError):
        tcache = TrainCache.load(project_name, LSTM, postfix)
    tcaches = TrainCache.load_multiple(project_name, LSTM, postfix)
    assert len(tcaches) == 2

"""
# this test uses not fake data;
# before running this test, must run attractor2d.py
def test_lstm_with_cmds():
    project_name = 'attractor2d'
    chunk = CommandDataChunk.load(project_name)
    dataset = AutoRegressiveDataset.from_chunk(chunk)
    n_seq, n_state = dataset.data[0].shape 
    model = LSTM(torch.device('cpu'), n_state)

    _train(project_name, model, dataset, LSTM, Config(n_epoch=10))
    tcache = TrainCache.load(project_name, LSTM)
    val_seq = [dic['total'] for dic in tcache.validate_loss_dict_seq]
    train_seq = [dic['total'] for dic in tcache.train_loss_dict_seq]
    assert val_seq[-1] < 0.6 * val_seq[0]
    assert train_seq[-1] < 0.6 * train_seq[0]
"""

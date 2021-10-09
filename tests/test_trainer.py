import torch
from torch.utils.data import random_split
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.dataset import ReconstructionDataset
from mimic.dataset import AutoRegressiveDataset
from mimic.trainer import Config
from mimic.trainer import train
from mimic.trainer import TrainCache

from test_datatypes import image_datachunk
from test_datatypes import image_datachunk_with_encoder

def _train(model, dataset):
    project_name = 'test'
    n_total = len(dataset)
    train_set, val_set =  random_split(dataset, [n_total-2, 2])
    tcache = TrainCache(project_name=project_name)
    config = Config(n_epoch=2)
    train(model, train_set, val_set, tcache=tcache, config=config)
    cache_loaded = tcache.load(project_name, ImageAutoEncoder)

def test_train(image_datachunk, image_datachunk_with_encoder):
    dataset = ReconstructionDataset.from_chunk(image_datachunk)
    model = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(3, 28, 28))
    _train(model, dataset)

    dataset2 = AutoRegressiveDataset.from_chunk(image_datachunk_with_encoder)
    n_seq, n_state = dataset2.data[0].shape 
    model2 = LSTM(torch.device('cpu'), n_state)
    _train(model2, dataset2)






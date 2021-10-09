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

def _train(project_name, model, dataset, model_type):
    n_total = len(dataset)
    train_set, val_set =  random_split(dataset, [n_total-2, 2])
    tcache = TrainCache[model_type](project_name=project_name)
    config = Config(n_epoch=2)
    train(model, train_set, val_set, tcache=tcache, config=config)

def test_train(image_datachunk, image_datachunk_with_encoder):
    project_name = 'test'
    dataset = ReconstructionDataset.from_chunk(image_datachunk)
    model = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(3, 28, 28))
    _train(project_name, model, dataset, ImageAutoEncoder)
    tcache = TrainCache[ImageAutoEncoder].load(project_name)
    assert isinstance(tcache.best_model, ImageAutoEncoder)

    dataset2 = AutoRegressiveDataset.from_chunk(image_datachunk_with_encoder)
    n_seq, n_state = dataset2.data[0].shape 
    model2 = LSTM(torch.device('cpu'), n_state)
    _train(project_name, model2, dataset2, LSTM)
    tcache = TrainCache[LSTM].load(project_name)
    assert isinstance(tcache.best_model, LSTM)






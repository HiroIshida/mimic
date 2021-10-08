import torch
from torch.utils.data import random_split
from mimic.models import ImageAutoEncoder
from mimic.trainer import Config
from mimic.trainer import train
from mimic.trainer import TrainCache
from test_dataset import reconstruction_dataset

def test_train(reconstruction_dataset):
    ae = ImageAutoEncoder(16, torch.device('cpu'), image_shape=(3, 28, 28))
    n_total = len(reconstruction_dataset)
    train_set, val_set =  random_split(reconstruction_dataset, [n_total-2, 2])
    config = Config(n_epoch=2)
    train(ae, train_set, val_set, config=config)

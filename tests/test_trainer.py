import torch
from torch.utils.data import random_split
from mimic.models import ImageAutoEncoder
from mimic.dataset import ReconstructionDataset
from mimic.trainer import Config
from mimic.trainer import train
from mimic.trainer import TrainCache

from test_datatypes import image_datachunk

def test_train(image_datachunk):
    dataset = ReconstructionDataset.from_chunk(image_datachunk)
    project_name = 'test'
    ae = ImageAutoEncoder(16, torch.device('cpu'), image_shape=(3, 28, 28))
    n_total = len(dataset)
    train_set, val_set =  random_split(dataset, [n_total-2, 2])
    tcache = TrainCache(project_name=project_name)
    config = Config(n_epoch=2)
    train(ae, train_set, val_set, tcache=tcache, config=config)
    cache_loaded = tcache.load(project_name, ImageAutoEncoder)

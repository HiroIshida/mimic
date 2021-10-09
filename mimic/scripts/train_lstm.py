import argparse

import torch
from mimic.models.autoencoder import ImageAutoEncoder

from mimic.trainer import Config
from mimic.trainer import TrainCache
from mimic.trainer import train
from mimic.datatype import ImageDataChunk
from mimic.dataset import AutoRegressiveDataset
from mimic.models import LSTM
from mimic.scripts.utils import split_with_ratio
from mimic.scripts.utils import create_default_logger

def prepare_dataset(project_name: str, mode: str = 'image') -> AutoRegressiveDataset:
    if mode == 'image':
        tcache = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
        chunk = ImageDataChunk.load(project_name)
        chunk.set_encoder(tcache.best_model.encoder)
        dataset = AutoRegressiveDataset.from_chunk(chunk)
    return dataset

def train_lstm(project_name: str, config: Config) -> None:
    dataset = prepare_dataset(project_name)
    ds_train, ds_valid = split_with_ratio(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(device, dataset.n_state)
    tcache = TrainCache[LSTM](project_name)
    train(model, ds_train, ds_valid, tcache=tcache, config=config)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=1000, help='training epoch')
    args = parser.parse_args()
    project_name = args.pn
    n_epoch = args.n

    logger = create_default_logger(project_name)
    config = Config(n_epoch=n_epoch)
    train_lstm(project_name, config)

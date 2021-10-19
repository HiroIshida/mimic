import argparse
import typing
from typing import Union

import torch
from mimic.models.autoencoder import ImageAutoEncoder

from mimic.trainer import Config
from mimic.trainer import TrainCache
from mimic.trainer import train
from mimic.datatype import AbstractDataChunk
from mimic.datatype import ImageDataChunk
from mimic.datatype import ImageCommandDataChunk
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.models import LSTM
from mimic.models import DenseProp
from mimic.scripts.utils import split_with_ratio
from mimic.scripts.utils import create_default_logger

def prepare_chunk(project_name: str) -> AbstractDataChunk:
    tcache = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
    try:
        chunk: Union[ImageDataChunk, ImageCommandDataChunk] \
                = ImageCommandDataChunk.load(project_name)
    except FileNotFoundError:
        chunk = ImageDataChunk.load(project_name)
    chunk.set_encoder(tcache.best_model.get_encoder())
    return chunk

# TODO what is type of model_type. how to specify 'class' type??
# TODO Do type check! but this function is type-wise tricky...
@typing.no_type_check 
def train_propagator(project_name: str, model_type, config: Config) -> None:
    chunk = prepare_chunk(project_name)
    if model_type is LSTM:
        dataset = AutoRegressiveDataset.from_chunk(chunk)
    elif model_type is DenseProp:
        dataset = FirstOrderARDataset.from_chunk(chunk)
    else:
        raise RuntimeError
    ds_train, ds_valid = split_with_ratio(dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prop_model = model_type(device, dataset.n_state)
    tcache = TrainCache[model_type](project_name)
    train(prop_model, ds_train, ds_valid, tcache=tcache, config=config)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=1000, help='training epoch')
    parser.add_argument('--dense', action='store_true', help='use DenseProp instead of LSTM')
    args = parser.parse_args()
    project_name = args.pn
    n_epoch = args.n
    use_dense = args.dense

    logger = create_default_logger(project_name, 'propagator')
    config = Config(n_epoch=n_epoch)
    prop_model = DenseProp if use_dense else LSTM
    train_propagator(project_name, prop_model, config)

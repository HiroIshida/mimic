import argparse
import typing
from typing import Union

import torch

from mimic.trainer import Config
from mimic.trainer import TrainCache
from mimic.trainer import train
from mimic.datatype import AbstractDataChunk
from mimic.datatype import ImageDataChunk
from mimic.datatype import ImageCommandDataChunk
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import DenseProp
from mimic.models import ImageAutoEncoder
from mimic.models import BiasedDenseProp

from mimic.scripts.utils import split_with_ratio
from mimic.scripts.utils import create_default_logger
from mimic.scripts.utils import query_yes_no

def prepare_trained_image_chunk(project_name: str) -> AbstractDataChunk:
    tcache = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
    try:
        chunk_: Union[ImageDataChunk, ImageCommandDataChunk] \
                = ImageCommandDataChunk.load(project_name)
    except FileNotFoundError:
        chunk_ = ImageDataChunk.load(project_name)
    n_intact = 5
    _, chunk = chunk_.split(n_first=n_intact)
    chunk.set_encoder(tcache.best_model.get_encoder())
    return chunk

# TODO what is type of model_type. how to specify 'class' type??
# TODO Do type check! but this function is type-wise tricky...
@typing.no_type_check 
def train_propagator(project_name: str, model_type, config: Config) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chunk = prepare_trained_image_chunk(project_name)
    if model_type is LSTM:
        dataset = AutoRegressiveDataset.from_chunk(chunk)
        prop_model = LSTM(device, dataset.n_state)
    elif model_type is BiasedLSTM:
        dataset = BiasedAutoRegressiveDataset.from_chunk(chunk)
        prop_model = BiasedLSTM(device, dataset.n_state, dataset.n_bias)
    elif model_type is DenseProp:
        dataset = FirstOrderARDataset.from_chunk(chunk)
        prop_model = DenseProp(device, dataset.n_state)
    elif model_type is BiasedDenseProp:
        dataset = BiasedAutoRegressiveDataset.from_chunk(chunk)
        prop_model = BiasedDenseProp(device, dataset.n_state, dataset.n_bias)
    else:
        raise RuntimeError
    tcache = TrainCache[model_type](project_name, model_type)
    if tcache.exists_cache():
        if not query_yes_no('tcach exists. do you want to overwrite?'):
            raise RuntimeError('execution interrupt')
    ds_train, ds_valid = split_with_ratio(dataset)
    train(prop_model, ds_train, ds_valid, tcache=tcache, config=config)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=1000, help='training epoch')
    parser.add_argument('-model', type=str, default="lstm", help='model name')

    args = parser.parse_args()
    project_name = args.pn
    model_name = args.model
    n_epoch = args.n

    prop_model: type
    if model_name == 'lstm':
        prop_model = LSTM
    elif model_name == 'biased_lstm':
        prop_model = BiasedLSTM
    elif model_name == 'dense_prop':
        prop_model = DenseProp
    elif model_name == 'biased_dense_prop':
        prop_model = BiasedDenseProp
    else:
        raise RuntimeError('No such prop model named {} exists'.format(model_name))

    logger = create_default_logger(project_name, 'propagator_{}'.format(model_name))
    config = Config(n_epoch=n_epoch)
    train_propagator(project_name, prop_model, config)

import argparse
import typing
from typing import Union

import torch

from mimic.trainer import Config
from mimic.trainer import TrainCache
from mimic.trainer import train
from mimic.robot import KukaSpec
from mimic.datatype import FeatureInfo
from mimic.datatype import AbstractDataChunk
from mimic.datatype import ImageDataChunk
from mimic.datatype import ImageCommandDataChunk
from mimic.datatype import AugedImageCommandDataChunk
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import AugedAutoRegressiveDataset
from mimic.models import get_model_type_from_name
from mimic.models import LSTMConfig, BiasedLSTMConfig
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import DenseConfig, BiasedDenseConfig
from mimic.models import DenseProp
from mimic.models import ImageAutoEncoder
from mimic.models import BiasedDenseProp
from mimic.models import AugedLSTM, AugedLSTMConfig

from mimic.scripts.utils import split_with_ratio
from mimic.scripts.utils import create_default_logger
from mimic.scripts.utils import query_yes_no

def prepare_trained_image_chunk(project_name: str) -> AbstractDataChunk:
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
def train_propagator(project_name: str, model_type, config: Config, n_data_aug: int, cov_scale: float) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    chunk_ = prepare_trained_image_chunk(project_name)

    n_intact = 5
    _, chunk = chunk_.split(n_first=n_intact)
    if model_type is AugedLSTM:
        robot_spec =KukaSpec()
        chunk = AugedImageCommandDataChunk.from_imgcmd_chunk(chunk, robot_spec)
    finfo = chunk.get_feature_info()

    if model_type is LSTM:
        dataset = AutoRegressiveDataset.from_chunk(chunk, n_data_aug=n_data_aug, cov_scale=cov_scale)
        prop_model = LSTM(device, LSTMConfig.from_finfo(finfo))
    elif model_type is BiasedLSTM:
        dataset = BiasedAutoRegressiveDataset.from_chunk(chunk, n_data_aug=n_data_aug, cov_scale=cov_scale)
        prop_model = BiasedLSTM(device, BiasedLSTMConfig.from_finfo(finfo))
    elif model_type is DenseProp:
        dataset = AutoRegressiveDataset.from_chunk(chunk, n_data_aug=n_data_aug, cov_scale=cov_scale)
        prop_model = DenseProp(device, DenseConfig.from_finfo(finfo))
    elif model_type is BiasedDenseProp:
        dataset = BiasedAutoRegressiveDataset.from_chunk(chunk, n_data_aug=n_data_aug, cov_scale=cov_scale)
        prop_model = BiasedDenseProp(device, BiasedDenseConfig.from_finfo(finfo))
    elif model_type is AugedLSTM:
        dataset = AugedAutoRegressiveDataset.from_chunk(chunk, n_data_aug=n_data_aug, cov_scale=cov_scale)
        prop_model = AugedLSTM(device, AugedLSTMConfig.from_finfo(finfo, robot_spec))
    else:
        raise RuntimeError
    tcache = TrainCache[model_type](project_name, model_type)
    ds_train, ds_valid = split_with_ratio(dataset)
    train(prop_model, ds_train, ds_valid, tcache=tcache, config=config)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=1000, help='training epoch')
    parser.add_argument('-daug', type=int, default=0, help='number of data augmentation')
    parser.add_argument('-covscale', type=float, default=0.3, help='covariance scaler for data augmentation')
    parser.add_argument('-model', type=str, default="LSTM", help='model name')

    args = parser.parse_args()
    project_name = args.pn
    model_name = args.model
    n_epoch = args.n
    n_data_aug = args.daug
    cov_scale = args.covscale

    prop_model = get_model_type_from_name(model_name)

    logger = create_default_logger(project_name, 'propagator_{}'.format(model_name))
    config = Config(n_epoch=n_epoch)
    train_propagator(project_name, prop_model, config, n_data_aug, cov_scale)

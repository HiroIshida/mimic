import os
from dataclasses import dataclass
from functools import reduce
import operator
import copy
from torch._C import device

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import typing
from typing import Dict
from typing import List
from typing import Union
from typing import Optional
from typing import Generic
from typing import Type
from typing import Tuple
from typing import TypeVar
import logging
logger = logging.getLogger(__name__)

from mimic.models.common import _Model
from mimic.models.common import LossDictFloat
from mimic.models.common import to_scalar_values
from mimic.models.common import average_loss_dict
from mimic.file import dump_pickled_data
from mimic.file import _cache_name
from mimic.file import _cache_name_list
from mimic.file import load_pickled_data
from mimic.compat import is_compatible

@dataclass
class Config:
    batch_size: int = 200
    learning_rate : float = 0.001
    n_epoch : int = 1000

TrainCacheT = TypeVar('TrainCacheT', bound='TrainCache')
ModelT = TypeVar('ModelT', bound=_Model)
class TrainCache(Generic[ModelT]):
    project_name: str
    model_type: Type[ModelT] # TODO workaround. because python is not julia
    epoch: int
    train_loss_dict_seq: List[LossDictFloat]
    validate_loss_dict_seq: List[LossDictFloat]
    best_model: ModelT
    latest_model: ModelT
    cache_postfix: str

    def __init__(self, project_name: str, model_type: Type[ModelT], cache_postfix: Optional[str]=None):
        if cache_postfix is None:
            cache_postfix = ""
        self.project_name = project_name
        self.train_loss_dict_seq = []
        self.validate_loss_dict_seq = []
        self.cache_postfix = cache_postfix 
        self.model_type = model_type

    @typing.no_type_check
    def exists_cache(self) -> bool:
        filename_list = _cache_name_list(self.project_name, 
                self.__class__, self.model_type.__name__, self.cache_postfix)
        return len(filename_list)!=0

    def on_startof_epoch(self, epoch: int):
        logger.info('new epoch: {}'.format(epoch))
        self.epoch = epoch

    def on_train_loss(self, loss_dict: LossDictFloat, epoch: int):
        self.train_loss_dict_seq.append(loss_dict)
        logger.info('train_total_loss: {}'.format(loss_dict['total']))

    def on_validate_loss(self, loss_dict: LossDictFloat, epoch: int):
        self.validate_loss_dict_seq.append(loss_dict)
        logger.info('validate_total_loss: {}'.format(loss_dict['total']))

    def on_endof_epoch(self, model: ModelT, epoch: int):
        model = copy.deepcopy(model)
        model.to(torch.device('cpu'))
        self.latest_model = model

        totals = [dic['total'] for dic in self.validate_loss_dict_seq]
        min_loss = min(totals)
        if(totals[-1] == min_loss):
            self.best_model = model
            logger.info('model is updated')
        postfix = self.cache_postfix + model.hash_value
        dump_pickled_data(self, self.project_name, 
                self.best_model.__class__.__name__, postfix)

    def visualize(self, fax: Optional[Tuple]=None):
        fax = plt.subplots() if fax is None else fax
        fig, ax = fax
        train_loss_seq = [dic['total'] for dic in self.train_loss_dict_seq]
        valid_loss_seq = [dic['total'] for dic in self.validate_loss_dict_seq]
        ax.plot(train_loss_seq)
        ax.plot(valid_loss_seq)
        ax.set_yscale('log')
        ax.legend(['train', 'valid'])

    @classmethod
    def load(cls: Type[TrainCacheT], project_name: str, model_type: type, 
            cache_postfix: Optional[str]=None) -> TrainCacheT:
        # requiring "model_type" seems redundant but there is no way to 
        # use info of ModelT from @classmethod
        data_list = load_pickled_data(project_name, cls, model_type.__name__, cache_postfix)
        assert len(data_list) == 1, "data_list has {} elements.".format(len(data_list))
        return data_list[0]

    # TODO: probably has better design ...
    @classmethod
    def load_multiple(cls: Type[TrainCacheT], project_name: str, model_type: type, 
            cache_postfix: Optional[str]=None) -> List[TrainCacheT]:
        # requiring "model_type" seems redundant but there is no way to 
        # use info of ModelT from @classmethod
        data_list = load_pickled_data(project_name, cls, model_type.__name__, cache_postfix)
        assert len(data_list) > 1, "data_list has {} elements.".format(len(data_list))
        return data_list



def train(
        model: _Model, 
        dataset_train: Dataset,
        dataset_validate: Dataset,
        tcache: TrainCache,
        config: Config = Config()) -> None:

    logger.info('train start with config: {}'.format(config))
    assert is_compatible(model, dataset_train)
    assert is_compatible(model, dataset_validate)

    def move_to_device(sample):
        if isinstance(sample, torch.Tensor):
            return sample.to(model.device)
        elif isinstance(sample, list): # NOTE datalodaer return list type not tuple
            return tuple([e.to(model.device) for e in sample])
        else:
            raise RuntimeError

    train_loader = DataLoader(
            dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    validate_loader = DataLoader(
            dataset=dataset_validate, batch_size=config.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.put_on_device()
    for epoch in tqdm(range(config.n_epoch)):
        tcache.on_startof_epoch(epoch)

        model.train()
        train_ld_list : List[LossDictFloat] = []
        for samples in train_loader:
            optimizer.zero_grad()
            samples = move_to_device(samples)
            loss_dict = model.loss(samples)
            loss :torch.Tensor = reduce(operator.add, loss_dict.values())
            loss.backward()
            loss_dict['total'] = loss
            train_ld_list.append(to_scalar_values(loss_dict))
            optimizer.step()
        train_ld_sum = average_loss_dict(train_ld_list)
        tcache.on_train_loss(train_ld_sum, epoch)

        model.eval()
        validate_ld_list : List[LossDictFloat] = []
        for samples in validate_loader:
            samples = move_to_device(samples)
            loss_dict = model.loss(samples)
            loss_dict['total'] = reduce(operator.add, loss_dict.values())
            validate_ld_list.append(to_scalar_values(loss_dict))
        validate_ld_sum = average_loss_dict(validate_ld_list)
        tcache.on_validate_loss(validate_ld_sum, epoch)

        tcache.on_endof_epoch(model, epoch)

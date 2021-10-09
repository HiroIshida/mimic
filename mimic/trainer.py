from dataclasses import dataclass
from functools import reduce
import operator
import copy

from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import typing
from typing import Dict
from typing import List
from typing import Optional
from typing import Generic
from typing import Type
from typing import TypeVar
import logging
logger = logging.getLogger(__name__)

from mimic.models.common import _Model
from mimic.models.common import LossDictFloat
from mimic.models.common import to_scalar_values
from mimic.models.common import sum_loss_dict
from mimic.file import dump_pickled_data
from mimic.file import load_pickled_data

@dataclass
class Config:
    batch_size: int = 200
    learning_rate : float = 0.001
    n_epoch : int = 1000

TrainCacheT = TypeVar('TrainCacheT', bound='TrainCache')
ModelT = TypeVar('ModelT', bound=_Model)
class TrainCache(Generic[ModelT]):
    project_name: str
    train_loss_dict_seq: List[LossDictFloat]
    validate_loss_dict_seq: List[LossDictFloat]
    best_model: ModelT

    def __init__(self, project_name: str):
        self.project_name = project_name
        self.train_loss_dict_seq = []
        self.validate_loss_dict_seq = []

    def on_startof_epoch(self, epoch: int):
        logger.info('new epoch: {}'.format(epoch))

    def on_train_loss(self, loss_dict: LossDictFloat, epoch: int):
        self.train_loss_dict_seq.append(loss_dict)
        logger.info('train_total_loss: {}'.format(loss_dict['total']))

    def on_validate_loss(self, loss_dict: LossDictFloat, epoch: int):
        self.validate_loss_dict_seq.append(loss_dict)
        logger.info('validate_total_loss: {}'.format(loss_dict['total']))

    def on_endof_epoch(self, model: ModelT, epoch: int):
        totals = [dic['total'] for dic in self.validate_loss_dict_seq]
        min_loss = min(totals)
        if(totals[-1] == min_loss):
            model = copy.deepcopy(model)
            model.to(torch.device('cpu'))
            self.best_model = model
            logger.info('model is updated')
        dump_pickled_data(self, self.project_name, self.best_model.__class__.__name__)

    @classmethod
    def load(cls: Type[TrainCacheT], project_name: str, model_type: type) -> TrainCacheT:
        # requiring "model_type" seems redundant but there is no way to 
        # use info of ModelT from @classmethod
        return load_pickled_data(project_name, cls, model_type.__name__)

def train(
        model: _Model, 
        dataset_train: Dataset,
        dataset_validate: Dataset,
        tcache: TrainCache,
        config: Config = Config()) -> None:

    train_loader = DataLoader(
            dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    validate_loader = DataLoader(
            dataset=dataset_validate, batch_size=config.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.put_on_device()
    for epoch in tqdm(range(config.n_epoch)):

        model.train()
        train_ld_list : List[LossDictFloat] = []
        for samples in train_loader:
            optimizer.zero_grad()
            samples = samples.to(model.device)
            loss_dict = model.loss(samples)
            loss :torch.Tensor = reduce(operator.add, loss_dict.values())
            loss.backward()
            loss_dict['total'] = loss
            train_ld_list.append(to_scalar_values(loss_dict))
        train_ld_sum = sum_loss_dict(train_ld_list)
        tcache.on_train_loss(train_ld_sum, epoch)

        model.eval()
        validate_ld_list : List[LossDictFloat] = []
        for samples in validate_loader:
            samples = samples.to(model.device)
            loss_dict = model.loss(samples)
            loss_dict['total'] = reduce(operator.add, loss_dict.values())
            validate_ld_list.append(to_scalar_values(loss_dict))
        validate_ld_sum = sum_loss_dict(validate_ld_list)
        tcache.on_validate_loss(validate_ld_sum, epoch)

        tcache.on_endof_epoch(model, epoch)

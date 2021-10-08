from dataclasses import dataclass
from functools import reduce
import operator

from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Dict
from typing import List
from typing import Optional
import logging

from mimic.models.common import _Model
from mimic.models.common import LossDictNoGrad
from mimic.models.common import detach_clone
from mimic.models.common import sum_loss_dict

@dataclass
class Config:
    batch_size: int = 200
    learning_rate : float = 0.001
    n_epoch : int = 1000

class TrainCallback:
    def on_train_loss(self, loss_dict: LossDictNoGrad, epoch: int):
        pass # add logger

    def on_validate_loss(self, loss_dict: LossDictNoGrad, epoch: int):
        pass # add logger

    def on_endof_epoch(self, model: _Model, epoch: int):
        pass

def train(
        model: _Model, 
        callback: TrainCallback,
        dataset_train: Dataset,
        dataset_validate: Dataset,
        config: Config = Config()):

    train_loader = DataLoader(
            dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    validate_loader = DataLoader(
            dataset=dataset_validate, batch_size=config.batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    model.put_on_device()
    for epoch in tqdm(range(config.n_epoch)):

        model.train()
        train_ld_list : List[LossDictNoGrad] = []
        for samples in train_loader:
            optimizer.zero_grad()
            samples.to(model.device)
            loss_dict = model.loss(samples)
            loss :torch.Tensor = reduce(operator.add, loss_dict.values())
            loss.backward()
            train_ld_list.append(detach_clone(loss_dict))
        train_ld_sum = sum_loss_dict(train_ld_list)
        callback.on_train_loss(train_ld_sum, epoch)

        model.eval()
        validate_ld_list : List[LossDictNoGrad] = []
        for samples in validate_loader:
            samples.to(model.device)
            loss_dict = model.loss(samples)
            validate_ld_list.append(detach_clone(loss_dict))
        validate_ld_sum = sum_loss_dict(validate_ld_list)
        callback.on_validate_loss(validate_ld_sum, epoch)

        callback.on_endof_epoch(model, epoch)

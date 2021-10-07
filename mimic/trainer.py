from dataclasses import dataclass
from functools import reduce
import operator

from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from typing import Optional

from mimic.models.common import _Model

@dataclass
class Config:
    batch_size: int = 200
    learning_rate : float = 0.001
    n_epoch : int = 1000

class TrainCallback:
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
        for samples in train_loader:
            optimizer.zero_grad()
            samples.to(model.device)
            loss_dict = model.loss(samples)
            loss :torch.Tensor = reduce(operator.add, loss_dict.values())
            loss.backward()

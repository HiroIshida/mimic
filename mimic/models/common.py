from abc import ABC, abstractmethod, abstractclassmethod
import logging
logger = logging.getLogger(__name__)

import torch.nn as nn
import torch
from typing import Any, Optional, Type
from typing import Dict
from typing import List
from typing import NewType
from typing import TypeVar
from typing import Generic
from dataclasses import dataclass
import pickle
import hashlib

LossDict = NewType('LossDict', Dict[str, torch.Tensor])
LossDictFloat = NewType('LossDictFloat', Dict[str, float])

def to_scalar_values(ld: LossDict) -> LossDictFloat:
    ld_new = LossDictFloat({})
    for key in ld.keys():
        ld_new[key] = float(ld[key].detach().item())
    return ld_new

def average_loss_dict(loss_dict_list: List[LossDictFloat]) -> LossDictFloat:
    out = LossDictFloat({})
    keys = loss_dict_list[0].keys()
    for loss_dict in loss_dict_list:
        for key in keys:
            if key in out:
                out[key] += loss_dict[key]
            else:
                out[key] = loss_dict[key]

    for key in keys:
        out[key] /= len(loss_dict_list)
    return out

class _ModelConfigBase:
    @property
    def hash_value(self) -> str:
        if len(self.__dict__.keys()) == 0:
            return ""
        data_pickle = pickle.dumps(self)
        data_md5 = hashlib.md5(data_pickle).hexdigest()
        return data_md5[:7]

@dataclass
class NullConfig(_ModelConfigBase): ...

MConfigT = TypeVar('MConfigT', bound='_ModelConfigBase')
class _Model(nn.Module, ABC, Generic[MConfigT]):
    device : torch.device
    config: MConfigT
    def __init__(self, device: torch.device, config: MConfigT):
        super().__init__()
        self.device = device
        self.config = config
        logger.info('model name: {}'.format(self.__class__.__name__))
        logger.info('model config: {}'.format(config))
        logger.info('model is initialized')

    def put_on_device(self): self.to(self.device)

    @property
    def hash_value(self) -> str: return self.config.hash_value

    @abstractmethod
    def loss(self, sample : Any) -> LossDict: ...

    @abstractmethod
    def _create_layers(self, **kwargs) -> None: ...


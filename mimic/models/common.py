from abc import ABC, abstractmethod, abstractclassmethod
import logging
from torch._C import finfo
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

from mimic.datatype import FeatureInfo

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
        data_pickle = pickle.dumps(self)
        data_md5 = hashlib.md5(data_pickle).hexdigest()
        return data_md5[:7]

class _PropModelConfigBase(_ModelConfigBase):
    finfo: Optional[FeatureInfo] = None
    @property
    def n_img_feature(self): 
        if finfo is None: return None
        return self.finfo.n_img_feature
    @property
    def n_cmd_feature(self): 
        if finfo is None: return None
        return self.finfo.n_cmd_feature
    @property
    def n_aug_feature(self): 
        if finfo is None: return None
        return self.finfo.n_aug_feature

@dataclass
class NullConfig(_ModelConfigBase): ...

MConfigT = TypeVar('MConfigT', bound='_ModelConfigBase')
class _Model(nn.Module, ABC, Generic[MConfigT]):
    device : torch.device
    config: MConfigT
    def __init__(self, device: torch.device, config: MConfigT):
        super().__init__()

        assert isinstance(config, self.compat_modelconfig())

        self.device = device
        self.config = config
        logger.info('model name: {}'.format(self.__class__.__name__))
        logger.info('model config: {}'.format(config))
        logger.info('model is initialized')

    def put_on_device(self): self.to(self.device)

    @abstractclassmethod # python sucks...
    def compat_modelconfig(cls) -> Type[MConfigT]: ...

    @property
    def hash_value(self) -> str: return self.config.hash_value

    @abstractmethod
    def loss(self, sample : Any) -> LossDict: ...

    @abstractmethod
    def _create_layers(self, **kwargs) -> None: ...

PropConfigT = TypeVar('PropConfigT', bound='_PropModelConfigBase')
class _PropModel(_Model[PropConfigT]):
    finfo: Optional[FeatureInfo]
    def __init__(self, device: torch.device, config: PropConfigT, finfo: Optional[FeatureInfo]=None):
        super().__init__(device, config)
        self.finfo = finfo

    def has_feature_info(self) -> bool:
        return self.config.finfo != None

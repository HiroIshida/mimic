from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import Any
from typing import Dict
from typing import List
from typing import NewType

LossDict = NewType('LossDict', Dict[str, torch.Tensor])
LossDictFloat = NewType('LossDictFloat', Dict[str, float])

def to_scalar_values(ld: LossDict) -> LossDictFloat:
    ld_new = LossDictFloat({})
    for key in ld.keys():
        ld_new[key] = float(ld[key].detach().item())
    return ld_new

def sum_loss_dict(loss_dict_list: List[LossDictFloat]) -> LossDictFloat:
    out = LossDictFloat({})
    keys = loss_dict_list[0].keys()
    for loss_dict in loss_dict_list:
        for key in keys:
            if key in out:
                out[key] += loss_dict[key]
            else:
                out[key] = loss_dict[key]
    return out

class _Model(nn.Module, ABC):
    device : torch.device
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def put_on_device(self): self.to(self.device)

    @abstractmethod
    def loss(self, sample : Any) -> LossDict: ...

    @abstractmethod
    def _create_layers(self, **kwargs) -> None: ...


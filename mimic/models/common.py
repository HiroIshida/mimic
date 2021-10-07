from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import Any
from typing import Dict

class _Model(nn.Module, ABC):
    device : torch.device
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def put_on_device(self): self.to(self.device)

    @abstractmethod
    def loss(self, sample : Any) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def _create_layers(self, **kwargs) -> None: ...


from typing import List
from typing import Tuple
import torch
from torch._C import device
import torch.nn as nn

from mimic.models.common import _Model
from mimic.models.common import LossDict
from mimic.dataset import FirstOrderARDataset

def create_linear_layer(n_input, n_output, n_hidden, n_layer) -> nn.Module:
    layers = []
    input_layer = nn.Linear(n_input, n_hidden)
    layers.append(input_layer)
    for _ in range(n_layer - 1):
        middle_layer = nn.Linear(n_hidden, n_hidden)
        layers.append(middle_layer)
    output_layer = nn.Linear(n_hidden, n_output)
    layers.append(output_layer)
    layer = nn.Sequential(*layers)
    return layer

class DenseProp(_Model):
    n_state: int
    n_hidden: int
    n_layer: int
    layer: nn.Module
    def __init__(self, 
            device: device, 
            n_state: int, 
            n_hidden: int=200, 
            n_layer: int=2):
        assert n_layer > 0
        _Model.__init__(self, device)
        self.n_state = n_state
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self._create_layers()

    def _create_layers(self, **kwargs) -> None:
        self.layer = create_linear_layer(
                self.n_state, self.n_state, self.n_hidden, self.n_layer)

    def forward(self, sample_pre: torch.Tensor):
        return self.layer(sample_pre)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> LossDict:
        sample_pre, sample_post = sample
        post_pred = self.forward(sample_pre)
        loss_value = nn.MSELoss()(post_pred, sample_post)
        return LossDict({'prediction': loss_value})

class BiasedDenseProp(_Model):
    n_state: int
    n_hidden: int
    n_layer: int
    layer: nn.Module
    n_bias: int
    def __init__(self, 
            device: device, 
            n_state: int, 
            n_bias: int,
            n_hidden: int=200, 
            n_layer: int=2):
        _Model.__init__(self, device)
        self.n_state = n_state
        self.n_bias = n_bias
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self._create_layers()

    def _create_layers(self, **kwargs) -> None:
        self.layer = create_linear_layer(
                self.n_state + self.n_bias, self.n_state, self.n_hidden, self.n_layer)

    def forward(self, sample_pre: torch.Tensor, bias: torch.Tensor):
        assert sample_pre.ndim == 2 and bias.ndim == 2
        assert sample_pre.shape[0] == bias.shape[0]
        sample_pre_cat = torch.cat((sample_pre, bias), dim=1)
        return self.layer(sample_pre_cat)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> LossDict:
        sample_pre, bias, sample_post = sample
        post_pred = self.forward(sample_pre, bias)
        loss_value = nn.MSELoss()(post_pred, sample_post)
        return LossDict({'prediction': loss_value})

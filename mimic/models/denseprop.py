from typing import List
from typing import Tuple
import torch
from torch._C import device
import torch.nn as nn

from mimic.models.common import _Model
from mimic.models.common import LossDict
from mimic.dataset import FirstOrderARDataset

class DenseProp(_Model):
    n_state: int
    n_input: int
    n_output: int
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
        self.n_input = n_state
        self.n_output = n_state
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self._create_layers()

    def _create_layers(self, **kwargs) -> None:
        layers = []
        input_layer = nn.Linear(self.n_input, self.n_hidden)
        layers.append(input_layer)
        for _ in range(self.n_layer - 1):
            middle_layer = nn.Linear(self.n_hidden, self.n_hidden)
            layers.append(middle_layer)
        output_layer = nn.Linear(self.n_hidden, self.n_output)
        layers.append(output_layer)
        self.layer = nn.Sequential(*layers)

    def forward(self, sample_pre: torch.Tensor):
        return self.layer(sample_pre)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor]) -> LossDict:
        sample_pre, sample_post = sample
        post_pred = self.forward(sample_pre)
        loss_value = nn.MSELoss()(post_pred, sample_post)
        return LossDict({'prediction': loss_value})

class BiasedDenseProp(DenseProp):
    b_bias: int
    def __init__(self, 
            device: device, 
            n_state: int, 
            n_bias: int,
            n_hidden: int=200, 
            n_layer: int=2):
        _Model.__init__(self, device)
        self.n_state = n_state
        self.n_bias = n_bias
        self.n_input = n_state + n_bias
        self.n_output = n_bias
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self._create_layers()

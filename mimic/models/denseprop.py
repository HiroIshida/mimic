from typing import List
from typing import Tuple
from typing import Optional
import torch
from torch._C import device
import torch.nn as nn

from mimic.models.common import _Model
from mimic.models.common import LossDict
from mimic.dataset import FirstOrderARDataset

def create_linear_layers(n_input, n_output, n_hidden, n_layer) -> List[nn.Linear]:
    layers = []
    input_layer = nn.Linear(n_input, n_hidden)
    layers.append(input_layer)
    for _ in range(n_layer):
        middle_layer = nn.Linear(n_hidden, n_hidden)
        layers.append(middle_layer)
    output_layer = nn.Linear(n_hidden, n_output)
    layers.append(output_layer)
    return layers

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
        layers = create_linear_layers(
                self.n_state, self.n_state, self.n_hidden, self.n_layer)
        self.layer = nn.Sequential(*layers)

    def forward(self, sample_pre: torch.Tensor):
        return self.layer(sample_pre)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor], 
            state_slicer: Optional[slice]=None, reduction='mean') -> LossDict:
        if state_slicer is None:
            state_slicer = slice(None)
        assert state_slicer.step == None
        sample_pre, sample_post = sample
        post_pred = self.forward(sample_pre)
        loss_value = nn.MSELoss(reduction=reduction)(post_pred[:, state_slicer], sample_post[:, state_slicer])
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
        layers = create_linear_layers(
                self.n_state + self.n_bias, self.n_state, self.n_hidden, self.n_layer)
        self.layer = nn.Sequential(*layers)

    def forward(self, sample_pre: torch.Tensor, bias: torch.Tensor):
        sample_pre_cat = torch.cat((sample_pre, bias), dim=1)
        return self.layer(sample_pre_cat)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
            state_slicer: Optional[slice] = None, reduction='mean') -> LossDict:
        if state_slicer is None:
            state_slicer = slice(None)
        assert state_slicer.step == None
        sample_pre, sample_post, bias = sample
        post_pred = self.forward(sample_pre, bias)
        loss_value = nn.MSELoss(reduction=reduction)(post_pred[:, state_slicer], sample_post[:, state_slicer])
        return LossDict({'prediction': loss_value})

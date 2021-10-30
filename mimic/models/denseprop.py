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
    # TODO shold be part of DenseProp class
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

    def forward(self, sample_input: torch.Tensor):
        n_batch, n_seq, n_input = sample_input.shape
        assert n_input == self.n_state + self.n_bias

        init_feature = torch.unsqueeze(sample_input[:, 0, self.n_bias:], dim=1)
        init_bias_feature = torch.unsqueeze(sample_input[:, 0, :self.n_bias], dim=1)

        pred_feature_list = []
        feature = init_feature
        for i in range(n_seq):
            feature_cat = torch.cat((init_bias_feature, feature), dim=2)
            feature = self.layer(feature_cat)
            pred_feature_list.append(feature)
        return torch.cat(pred_feature_list, dim=1)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor],
            state_slicer: Optional[slice] = None, reduction='mean') -> LossDict:

        sample_input, sample_output = sample
        n_batch, n_seq, n_input = sample_input.shape
        n_batch2, n_seq2, n_output = sample_output.shape
        assert n_batch == n_batch2 
        assert n_seq == n_seq2
        assert n_input == self.n_state + self.n_bias, 'expect: {}, got: {}'.format(self.n_state + self.n_bias, n_input)
        assert n_output == self.n_state, 'expect: {}, got: {}'.format(self.n_state, n_output)

        pred_output = self.forward(sample_input)
        loss_value = nn.MSELoss(reduction=reduction)(pred_output[:, state_slicer], sample_output[:, state_slicer])
        return LossDict({'prediction': loss_value})

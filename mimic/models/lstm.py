import logging
from typing import Optional
from typing import Tuple
logger = logging.getLogger(__name__)

import torch
from torch._C import device
import torch.nn as nn

from mimic.models.common import _Model
from mimic.models.common import LossDict
from mimic.dataset import AutoRegressiveDataset

class LSTM(_Model):
    """
    Note that n_state 
    """
    n_flag: int = 1
    n_state: int
    n_hidden: int
    n_layer: int
    n_bias: int
    lstm_layer: nn.LSTM
    output_layer: nn.Linear

    def __init__(self, device: device, n_state: int, n_bias: int=0, n_hidden: int=200, n_layer: int=2):
        _Model.__init__(self, device)
        self.n_state = n_state
        self.n_bias = n_bias
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self._create_layers()

    def _create_layers(self, **kwargs) -> None:
        n_input = self.n_state + self.n_bias
        self.lstm_layer = nn.LSTM(n_input, self.n_hidden, self.n_layer, batch_first=True)
        self.output_layer = nn.Linear(self.n_hidden, self.n_state)

    def forward(self, sample_input: torch.Tensor) -> torch.Tensor:
        n_batch, n_seq, n_input = sample_input.shape
        assert n_input == self.n_state + self.n_bias

        lstm_out, _ = self.lstm_layer(sample_input)
        out = self.output_layer(lstm_out)
        return out

    def loss(self, samples: Tuple[torch.Tensor, torch.Tensor], state_slicer: Optional[slice] = None, reduction='mean') -> LossDict:
        sample_input, sample_output = samples
        n_batch, n_seq, n_input = sample_input.shape
        n_batch2, n_seq2, n_output = sample_output.shape
        assert n_batch == n_batch2 
        assert n_seq == n_seq2
        assert n_input == self.n_state + self.n_bias, 'expect: {}, got: {}'.format(self.n_state + self.n_bias, n_input)
        assert n_output == self.n_state, 'expect: {}, got: {}'.format(self.n_state, n_output)

        if state_slicer is None:
            state_slicer = slice(None)
        assert state_slicer.step == None

        pred_output = self.forward(sample_input)
        loss_value = nn.MSELoss(reduction=reduction)(pred_output[:, state_slicer], sample_output[:, state_slicer])
        return LossDict({'prediction': loss_value})

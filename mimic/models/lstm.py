import logging
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
    lstm_layer: nn.LSTM
    output_layer: nn.Linear

    def __init__(self, device: device, n_state: int, n_hidden: int=200, n_layer: int=2):
        _Model.__init__(self, device)
        self.n_state = n_state
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self._create_layers()

    def _create_layers(self, **kwargs) -> None:
        self.lstm_layer = nn.LSTM(self.n_state, self.n_hidden, self.n_layer, batch_first=True)
        self.output_layer = nn.Linear(self.n_hidden, self.n_state)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        n_batch, n_seq, n_state = sample.shape

        lstm_out, _ = self.lstm_layer(sample)
        out = self.output_layer(lstm_out)
        return out

    def loss(self, sample: torch.Tensor) -> LossDict:
        seq_feed, seq_pred_gt = sample[:, :-1, :], sample[:, 1:, :]
        seq_pred = self.forward(seq_feed)
        loss_value = nn.MSELoss()(seq_pred, seq_pred_gt)
        return LossDict({'prediction': loss_value})

import torch
from torch._C import device
import torch.nn as nn

from mimic.models.common import _Model
from mimic.models.common import LossDict

class LSTM(_Model):
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
        self._create_layers(n_state, n_hidden, n_layer)

    def _create_layers(self, n_state, n_hidden, n_layer) -> None:
        self.lstm_layer = nn.LSTM(n_state, n_hidden, n_layer, batch_first=True)
        self.output_layer = nn.Linear(n_hidden, n_state)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm_layer(sample)
        return self.output_layer(out)

    def loss(self, sample: torch.Tensor) -> LossDict:
        seq_feed, seq_pred_gt = sample[:, :-1, :], sample[:, 1:, :]
        seq_pred = self.forward(seq_feed)
        loss_value = nn.MSELoss()(seq_feed, seq_pred)
        return LossDict({'prediction': loss_value})

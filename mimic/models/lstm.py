from abc import ABC, abstractproperty
import logging
from typing import Optional
from typing import Tuple
from typing import TypeVar
from dataclasses import dataclass
logger = logging.getLogger(__name__)

import torch
from torch._C import device
import torch.nn as nn

from mimic.datatype import FeatureInfo
from mimic.robot import RobotSpecBase
from mimic.models.common import _PropModel, NullConfig, _ModelConfigBase, _PropModelConfigBase
from mimic.models.common import LossDict
from mimic.dataset import AutoRegressiveDataset

@dataclass
class LSTMConfig(_PropModelConfigBase):
    n_state: int
    n_hidden: int = 200
    n_layer: int = 2

    @classmethod
    def from_finfo(cls, finfo: FeatureInfo, **kwargs) -> 'LSTMConfig': 
        obj = cls(finfo.n_img_feature + finfo.n_cmd_feature + 1, **kwargs)
        obj.finfo = finfo
        return obj

    @property
    def n_bias(self) -> int: return 0
    @property
    def n_aug(self) -> int: return 0

@dataclass
class BiasedLSTMConfig(_PropModelConfigBase):
    n_state: int
    n_bias: int
    n_hidden: int = 200
    n_layer: int = 2

    @classmethod
    def from_finfo(cls, finfo: FeatureInfo, **kwargs) -> 'BiasedLSTMConfig': 
        obj = cls(finfo.n_cmd_feature + 1, finfo.n_img_feature, **kwargs)
        obj.finfo = finfo
        return obj

    @property
    def n_aug(self) -> int: return 0

@dataclass
class AugedLSTMConfig(_PropModelConfigBase):
    n_state: int
    n_aug: int
    robot_spec: RobotSpecBase
    n_hidden: int = 200
    n_layer: int = 2

    @classmethod
    def from_finfo(cls, finfo: FeatureInfo, robot_spec: RobotSpecBase, **kwargs) -> 'AugedLSTMConfig': 
        obj = cls(
                finfo.n_cmd_feature + finfo.n_img_feature + 1, 
                finfo.n_aug_feature, 
                robot_spec,
                **kwargs)
        obj.finfo = finfo
        return obj

    @property
    def n_bias(self) -> int: return 0

LSTMConfigT = TypeVar('LSTMConfigT', bound=_PropModelConfigBase)
class LSTMBase(_PropModel[LSTMConfigT]):
    """
    Note that n_state 
    """
    n_flag: int = 1
    lstm_layer: nn.LSTM
    output_layer: nn.Linear

    @property
    def n_state(self) -> int: return self.config.n_state # type: ignore
    @property
    def n_bias(self) -> int: return self.config.n_bias # type: ignore
    @property
    def n_aug(self) -> int: return self.config.n_aug # type: ignore
    @property
    def n_hidden(self) -> int: return self.config.n_hidden # type: ignore
    @property
    def n_layer(self) -> int: return self.config.n_layer # type: ignore

    def __init__(self, device: device, config: LSTMConfigT):
        _PropModel.__init__(self, device, config)
        self._create_layers()

    def _create_layers(self, **kwargs) -> None:
        n_input = self.n_state + self.n_aug + self.n_bias
        n_output =  self.n_state + self.n_aug
        self.lstm_layer = nn.LSTM(n_input, self.n_hidden, self.n_layer, batch_first=True)
        self.output_layer = nn.Linear(self.n_hidden, n_output)

    def forward(self, sample_input: torch.Tensor) -> torch.Tensor:
        n_batch, n_seq, n_input = sample_input.shape
        n_input_expect = self.n_state + self.n_aug + self.n_bias
        assert n_input == n_input_expect

        lstm_out, _ = self.lstm_layer(sample_input)
        out = self.output_layer(lstm_out)
        return out

    def loss(self, samples: Tuple[torch.Tensor, torch.Tensor], state_slicer: Optional[slice] = None, reduction='mean') -> LossDict:
        sample_input, sample_output = samples
        n_batch, n_seq, n_input = sample_input.shape
        n_batch2, n_seq2, n_output = sample_output.shape
        n_input_expect = self.n_state + self.n_aug + self.n_bias
        n_output_expect = self.n_state + self.n_aug
        assert n_batch == n_batch2 
        assert n_seq == n_seq2
        assert n_input == n_input_expect, 'expect: {}, got: {}'.format(n_input_expect, n_input)
        assert n_output == n_output_expect, 'expect: {}, got: {}'.format(n_output_expect, n_output)

        if state_slicer is None:
            state_slicer = slice(None)
        assert state_slicer.step == None

        pred_output = self.forward(sample_input)
        loss_value = nn.MSELoss(reduction=reduction)(pred_output[:, state_slicer], sample_output[:, state_slicer])
        return LossDict({'prediction': loss_value})

class LSTM(LSTMBase[LSTMConfig]):
    def __init__(self, device: device, config: LSTMConfig):
        super().__init__(device, config)

    @classmethod
    def compat_modelconfig(cls): return LSTMConfig

class BiasedLSTM(LSTMBase):
    def __init__(self, device: device, config: BiasedLSTMConfig):
        super().__init__(device, config)

    @classmethod
    def compat_modelconfig(cls): return BiasedLSTMConfig

class AugedLSTM(LSTMBase):
    def __init__(self, device: device, config: AugedLSTMConfig):
        super().__init__(device, config)

    @classmethod
    def compat_modelconfig(cls): return AugedLSTMConfig

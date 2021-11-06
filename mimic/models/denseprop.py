from abc import ABC, abstractproperty
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Optional
from typing import Type
from dataclasses import dataclass
import torch
from torch._C import device
import torch.nn as nn
from torch.nn.modules import activation

from mimic.models.common import _Model, NullConfig, _ModelConfigBase
from mimic.models.common import LossDict
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import KinematicsDataset
from mimic.robot import RobotSpecBase

@dataclass
class DenseConfig(_ModelConfigBase):
    n_state: int
    n_hidden: int = 200
    n_layer: int = 2
    activation: Optional[str] = None
    @property
    def n_bias(self) -> int: return 0

@dataclass
class BiasedDenseConfig(_ModelConfigBase):
    n_state: int
    n_bias: int
    n_hidden: int = 200
    n_layer: int = 2
    activation: Optional[str] = None

def create_linear_layers(n_input, n_output, n_hidden, n_layer,
        activation: Optional[str]) -> List[nn.Module]:

    AT: Optional[Type[nn.Module]] = None
    if activation=='relu':
        AT = nn.ReLU
    elif activation=='sigmoid':
        AT = nn.Sigmoid
    elif activation=='tanh':
        AT = nn.Tanh

    layers: List[nn.Module] = []
    input_layer = nn.Linear(n_input, n_hidden)
    layers.append(input_layer)
    if AT is not None:
        layers.append(AT())

    for _ in range(n_layer):
        middle_layer = nn.Linear(n_hidden, n_hidden)
        layers.append(middle_layer)
        if AT is not None:
            layers.append(AT())

    output_layer = nn.Linear(n_hidden, n_output)
    layers.append(output_layer)
    return layers

DenseConfigT = TypeVar('DenseConfigT', bound=_ModelConfigBase)
class DenseBase(_Model[DenseConfigT]):
    # TODO shold be part of DenseProp class
    layer: nn.Module
    def __init__(self, 
            device: device, 
            config: DenseConfigT):
        _Model.__init__(self, device, config)
        self._create_layers()

    @property
    def n_state(self) -> int: return self.config.n_state # type: ignore
    @property
    def n_bias(self) -> int: return self.config.n_bias # type: ignore
    @property
    def n_hidden(self) -> int: return self.config.n_hidden # type: ignore
    @property
    def n_layer(self) -> int: return self.config.n_layer # type: ignore
    @property
    def activation(self) -> Optional[str]: return self.config.activation # type: ignore

    def _create_layers(self, **kwargs) -> None:
        layers = create_linear_layers(
                self.n_state + self.n_bias, 
                self.n_state, 
                self.n_hidden, 
                self.n_layer, 
                self.activation)
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


class DenseProp(DenseBase):
    def __init__(self, device: device, config: DenseConfig):
        assert isinstance(config, DenseConfig)
        super().__init__(device, config)

class DeprecatedDenseProp(DenseBase):
    def __init__(self, device: device, config: DenseConfig):
        assert isinstance(config, DenseConfig)
        super().__init__(device, config)

    # override!!
    def forward(self, sample_pre: torch.Tensor):
        return self.layer(sample_pre)

    # override!!
    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor], 
            state_slicer: Optional[slice]=None, reduction='mean') -> LossDict:
        if state_slicer is None:
            state_slicer = slice(None)
        assert state_slicer.step == None
        sample_pre, sample_post = sample
        post_pred = self.forward(sample_pre)
        loss_value = nn.MSELoss(reduction=reduction)(post_pred[:, state_slicer], sample_post[:, state_slicer])
        return LossDict({'prediction': loss_value})

class BiasedDenseProp(DenseBase):
    def __init__(self, device: device, config: BiasedDenseConfig):
        assert isinstance(config, BiasedDenseConfig)
        super().__init__(device, config)

@dataclass
class KinemaNetConfig(_ModelConfigBase):
    n_hidden: int = 200
    n_layer: int = 6
    activation: Optional[str] = None

class KinemaNet(_Model[KinemaNetConfig]):
    robot_spec: RobotSpecBase
    n_input: int
    n_output: int
    layer: nn.Module
    def __init__(self, 
            device: device, 
            robot_spec: RobotSpecBase,
            config: KinemaNetConfig):
        assert isinstance(config, KinemaNetConfig)
        _Model.__init__(self, device, config)
        self.robot_spec = robot_spec
        self.n_input = robot_spec.n_joint
        self.n_output = robot_spec.n_out
        self._create_layers()

    def _create_layers(self, **kwargs) -> None:
        layers = create_linear_layers(
                self.n_input, 
                self.n_output, 
                self.config.n_hidden, 
                self.config.n_layer,
                self.config.activation
                )
        self.layer = nn.Sequential(*layers)

    def forward(self, sample: torch.Tensor): return self.layer(sample)

    def loss(self, sample: Tuple[torch.Tensor, torch.Tensor]):
        sample_input, sample_output = sample
        pred_output = self.forward(sample_input)
        loss_value = nn.MSELoss()(pred_output, sample_output)
        return LossDict({'prediction': loss_value})

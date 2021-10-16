import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from typing import Tuple
from typing import List
from typing import TypeVar
from typing import Generic
from mimic.dataset import AutoRegressiveDataset
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from abc import ABC, abstractmethod

def attach_flag(vec: torch.Tensor) -> torch.Tensor: 
    flag = torch.tensor([AutoRegressiveDataset.continue_flag])
    return torch.cat((vec, flag))

def strip_flag(vec: torch.Tensor) -> torch.Tensor: return vec[:-1]

def force_continue_flag(vec: torch.Tensor) -> None: 
    vec[-1] = AutoRegressiveDataset.continue_flag

StateT = TypeVar('StateT') # TODO maybe this is unncessarly
class AbstractPredictor(ABC, Generic[StateT]):
    propagator: LSTM
    states: List[torch.Tensor]
    def __init__(self, propagator: LSTM):
        self.propagator = propagator
        self.states = []

    def _lstm_predict(self, n_horizon: int, with_feeds: bool=False) -> List[torch.Tensor]:
        feeds = copy.deepcopy(self.states)
        preds: List[torch.Tensor] = []
        for _ in range(n_horizon):
            states = torch.stack(feeds)
            tmp = self.propagator(torch.unsqueeze(states, 0))
            out = torch.squeeze(tmp, dim=0)[-1].detach().clone()
            force_continue_flag(out)
            feeds.append(out)
            preds.append(out)
        return feeds if with_feeds else preds

    @abstractmethod
    def feed(self, state: StateT) -> None: ...
    @abstractmethod
    def predict(self, n_horizon: int, with_feeds: bool) -> List[StateT]: ...

class LSTMPredictor(AbstractPredictor[np.ndarray]):

    def feed(self, cmd: np.ndarray) -> None: 
        assert cmd.ndim == 1
        cmd_torch = torch.from_numpy(cmd).float()
        cmd_with_flag = attach_flag(cmd_torch)
        self.states.append(cmd_with_flag)

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[np.ndarray]:
        raw_preds = self._lstm_predict(n_horizon, with_feeds)
        raw_preds_stripped = [strip_flag(pred) for pred in raw_preds]
        preds_np = [pred.detach().numpy() for pred in raw_preds_stripped]
        return preds_np

class ImageLSTMPredictor(AbstractPredictor[np.ndarray]):
    auto_encoder: ImageAutoEncoder

    def __init__(self, propagator: LSTM, auto_encoder: ImageAutoEncoder):
        assert auto_encoder.n_bottleneck == propagator.n_state - propagator.n_flag
        self.auto_encoder = auto_encoder
        super().__init__(propagator)

    def feed(self, img: np.ndarray) -> None:
        assert img.ndim == 3
        img_torch = torchvision.transforms.ToTensor()(img).float()
        feature_ = self.auto_encoder.encoder(torch.unsqueeze(img_torch, 0))
        feature = torch.squeeze(feature_.detach().clone(), dim=0)
        feature_with_flag = attach_flag(feature)
        self.states.append(feature_with_flag)

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[np.ndarray]:
        raw_preds = self._lstm_predict(n_horizon, with_feeds)
        raw_preds_stripped = [strip_flag(pred) for pred in raw_preds]
        image_preds = self.auto_encoder.decoder(torch.stack(raw_preds_stripped))
        lst = [np.array(torchvision.transforms.ToPILImage()(img)) for img in image_preds]
        return lst

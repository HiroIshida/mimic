import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from typing import Tuple
from typing import List
from typing import TypeVar
from typing import Generic
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from abc import ABC, abstractmethod

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
            out = torch.squeeze(tmp)[-1]
            feeds.append(out)
            preds.append(out)
        return feeds if with_feeds else preds

    @abstractmethod
    def feed(self, state: StateT) -> None: ...
    @abstractmethod
    def predict(self, n_horizon: int, with_feeds: bool) -> List[StateT]: ...

class SimplePredictor(AbstractPredictor[np.ndarray]):

    def feed(self, cmd: np.ndarray) -> None: 
        self.states.append(torch.from_numpy(cmd).float())

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[np.ndarray]:
        return [np.array(pred) for pred in self._lstm_predict(n_horizon, with_feeds)]

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
        feature = feature_.detach().clone()
        self.states.append(torch.squeeze(feature))

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[np.ndarray]:
        preds = torch.stack(self._lstm_predict(n_horizon, with_feeds))
        image_preds = self.auto_encoder.decoder(preds)
        lst = [np.array(torchvision.transforms.ToPILImage()(img)) for img in image_preds]
        return lst

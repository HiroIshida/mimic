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

StateT = TypeVar('StateT')
class AbstractImageBasedPredictor(ABC, Generic[StateT]):
    # TODO create type for Encoder and Propagator
    # implement get_encoder(self) -> Encoder: to AutoEncoder class
    # to handle dimension mismatch related to end-of-epoch flag
    # each encoder and propagator has attribute n_bottleneck & n_state
    # if n_bottleneck + 1 == n_state, then predictor automatically add flag state 
    auto_encoder: ImageAutoEncoder
    propatator: LSTM
    states: List[torch.Tensor] # concat of encoded image feature and cmds
    def __init__(self, auto_encoder: ImageAutoEncoder, propagator: LSTM):
        self.auto_encoder = auto_encoder
        self.propatator = propagator
        self.states = []

    @abstractmethod
    def feed(self, state: StateT) -> None: ...
    @abstractmethod
    def predict(self, n_horizon: int) -> List[StateT]: ...

class ImageLSTMPredictor(AbstractImageBasedPredictor[np.ndarray]):

    def __init__(self, auto_encoder: ImageAutoEncoder, propagator: LSTM):
        assert auto_encoder.n_bottleneck == propagator.n_state - propagator.n_flag
        super().__init__(auto_encoder, propagator)

    def feed(self, img: np.ndarray) -> None:
        assert img.ndim == 3
        img_torch = torchvision.transforms.ToTensor()(img).float()
        feature_ = self.auto_encoder.encoder(torch.unsqueeze(img_torch, 0))
        feature = feature_.detach().clone()
        self.states.append(torch.squeeze(feature))

    def _lstm_predict(self, n_horizon: int) -> List[torch.Tensor]:
        feeds = copy.deepcopy(self.states)
        preds: List[torch.Tensor] = []
        for _ in range(n_horizon):
            states = torch.stack(feeds)
            tmp = self.propatator(torch.unsqueeze(states, 0))
            out = torch.squeeze(tmp)[-1]
            feeds.append(out)
            preds.append(out)
        return preds

    def predict(self, n_horizon: int) -> List[np.ndarray]:
        preds = torch.stack(self._lstm_predict(n_horizon))
        image_preds = self.auto_encoder.decoder(preds)
        lst = [np.array(torchvision.transforms.ToPILImage()(img)) for img in image_preds]
        return lst

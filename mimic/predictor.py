import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from typing import Optional, Tuple
from typing import Union
from typing import List
from typing import TypeVar
from typing import Generic
from typing import NewType
from mimic.dataset import AutoRegressiveDataset
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.models import DenseProp
from mimic.models import BiasedDenseProp
from abc import ABC, abstractmethod

StateT = TypeVar('StateT') # TODO maybe this is unncessarly
FBPropT = TypeVar('FBPropT', bound=Union[LSTM, DenseProp])
FFPropT = TypeVar('FFPropT', bound=Union[BiasedDenseProp])
PropT = TypeVar('PropT', bound=Union[LSTM, DenseProp, BiasedDenseProp])

class AbstractPredictor(ABC, Generic[StateT, PropT]):
    propagator: PropT
    states: List[torch.Tensor]
    def __init__(self, propagator: PropT):
        self.propagator = propagator
        self.states = []

    def _predict(self, n_horizon: int, with_feeds: bool=False) -> List[torch.Tensor]:
        feeds = copy.deepcopy(self.states)
        preds: List[torch.Tensor] = []
        for _ in range(n_horizon):
            states = torch.stack(feeds)
            tmp = self.propagator(torch.unsqueeze(states, 0))
            out = torch.squeeze(tmp, dim=0)[-1].detach().clone()
            self._force_continue_flag_if_necessary(out)
            feeds.append(out)
            preds.append(out)
        raw_preds = feeds if with_feeds else preds
        return [self._strip_flag_if_necessary(e) for e in raw_preds]

    def _attach_flag_if_necessary(self, vec: torch.Tensor) -> torch.Tensor:
        if isinstance(self.propagator, LSTM):
            flag = torch.tensor([AutoRegressiveDataset.continue_flag])
            return torch.cat((vec, flag))
        return vec

    def _strip_flag_if_necessary(self, vec: torch.Tensor) -> torch.Tensor: 
        if isinstance(self.propagator, LSTM):
            return vec[:-1]
        return vec

    def _force_continue_flag_if_necessary(self, vec: torch.Tensor) -> None: 
        if isinstance(self.propagator, LSTM):
            vec[-1] = AutoRegressiveDataset.continue_flag

    def _feed(self, state: torch.Tensor) -> None:
        state_maybe_with_flag = self._attach_flag_if_necessary(state)
        self.states.append(state_maybe_with_flag)

    @abstractmethod
    def feed(self, state: StateT) -> None: ...

    @abstractmethod
    def predict(self, n_horizon: int, with_feeds: bool) -> List[StateT]: ...

class SimplePredictor(AbstractPredictor[np.ndarray, FBPropT]):

    def feed(self, cmd: np.ndarray) -> None: 
        assert cmd.ndim == 1
        cmd_torch = torch.from_numpy(cmd).float()
        return self._feed(cmd_torch)

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[np.ndarray]:
        preds = self._predict(n_horizon, with_feeds)
        preds_np = [pred.detach().numpy() for pred in preds]
        return preds_np

class ImagePredictor(AbstractPredictor[np.ndarray, FBPropT]):
    auto_encoder: ImageAutoEncoder

    def __init__(self, propagator: FBPropT, auto_encoder: ImageAutoEncoder):
        self.auto_encoder = auto_encoder
        super().__init__(propagator)

    def feed(self, img: np.ndarray) -> None:
        assert img.ndim == 3
        img_torch = torchvision.transforms.ToTensor()(img).float()
        feature_ = self.auto_encoder.encoder(torch.unsqueeze(img_torch, 0))
        feature = torch.squeeze(feature_.detach().clone(), dim=0)
        self._feed(feature)

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[np.ndarray]:
        preds = self._predict(n_horizon, with_feeds)
        image_preds = self.auto_encoder.decoder(torch.stack(preds))
        lst = [np.array(torchvision.transforms.ToPILImage()(img)) for img in image_preds]
        return lst

ImageCommandPair = Tuple[np.ndarray, np.ndarray]
class ImageCommandPredictor(AbstractPredictor[ImageCommandPair, FBPropT]):
    auto_encoder: ImageAutoEncoder

    def __init__(self, propagator: FBPropT, auto_encoder: ImageAutoEncoder):
        self.auto_encoder = auto_encoder
        super().__init__(propagator)

    def feed(self, imgcmd: ImageCommandPair) -> None:
        img, cmd = imgcmd
        assert img.ndim == 3 and cmd.ndim == 1
        img_torch = torchvision.transforms.ToTensor()(img).float()
        cmd_torch = torch.from_numpy(cmd).float()
        img_feature_ = self.auto_encoder.encoder(torch.unsqueeze(img_torch, 0))
        img_feature = torch.squeeze(img_feature_.detach().clone(), dim=0)
        feature = torch.cat((img_feature, cmd_torch))
        self._feed(feature)

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[ImageCommandPair]:
        n_img_feature = self.auto_encoder.n_bottleneck

        preds = self._predict(n_horizon, with_feeds)
        img_features, cmd_features\
                = zip(*[(e[:n_img_feature], e[n_img_feature:]) for e in preds])
        image_preds = self.auto_encoder.decoder(torch.stack(img_features))
        img_list = [np.array(torchvision.transforms.ToPILImage()(img)) for img in image_preds]
        cmd_list = [e.detach().numpy() for e in cmd_features]
        return list(zip(img_list, cmd_list))

MaybeNoneImageCommandPair = Tuple[Optional[np.ndarray], np.ndarray]
class FFImageCommandPredictor(AbstractPredictor[MaybeNoneImageCommandPair, FFPropT]):
    # TODO(HiroIShida) this class is so similar to ImageCommandPredictor, maybe we should make base class?
    auto_encoder: ImageAutoEncoder
    img_torch_one_shot: Optional[torch.Tensor]

    def __init__(self, propagator: FFPropT, auto_encoder: ImageAutoEncoder):
        super().__init__(propagator)
        self.auto_encoder = auto_encoder
        self.img_torch_one_shot = None

    def feed(self, imgcmd: MaybeNoneImageCommandPair) -> None:
        img, cmd = imgcmd
        assert cmd.ndim == 1
        if self.img_torch_one_shot is None:
            assert img is not None
            assert img.ndim == 3
            img_torch = torchvision.transforms.ToTensor()(img).float()
            img_feature_ = self.auto_encoder.encoder(torch.unsqueeze(img_torch, 0))
            img_feature = torch.squeeze(img_feature_.detach().clone(), dim=0)
            self.img_torch_one_shot = img_feature
        else:
            img_feature = self.img_torch_one_shot

        cmd_torch = torch.from_numpy(cmd).float()
        self._feed(cmd_torch)

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[MaybeNoneImageCommandPair]:
        preds = self._predict(n_horizon, with_feeds)
        cmd_list = [e.detach().numpy() for e in preds]
        return [(None, cmd) for cmd in cmd_list]

    # override!
    def _predict(self, n_horizon: int, with_feeds: bool=False) -> List[torch.Tensor]:
        feeds = copy.deepcopy(self.states)
        preds: List[torch.Tensor] = []
        for _ in range(n_horizon):
            states = torch.stack(feeds)
            assert self.img_torch_one_shot is not None # this required for mypy check
            bias = self.img_torch_one_shot.unsqueeze(0)
            bias_repeated = bias.expand(len(states), -1)
            tmp = self.propagator(states, bias_repeated)
            out = torch.squeeze(tmp, dim=0)[-1].detach().clone()
            self._force_continue_flag_if_necessary(out)
            feeds.append(out)
            preds.append(out)
        raw_preds = feeds if with_feeds else preds
        return [self._strip_flag_if_necessary(e) for e in raw_preds]

def get_model_specific_state_slice(autoencoder: ImageAutoEncoder, propagator: PropT) -> slice:
    idx_start = autoencoder.n_bottleneck
    idx_end = None
    if isinstance(propagator, LSTM): 
        idx_end = -1
    return slice(idx_start, idx_end)

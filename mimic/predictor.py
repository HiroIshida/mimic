import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import typing
from typing import Optional, Tuple
from typing import Union
from typing import List
from typing import TypeVar
from typing import Generic
from typing import NewType
from mimic.datatype import CommandDataSequence, ImageDataSequence, ImageCommandDataChunk
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import _DatasetFromChunk
from mimic.dataset import _continue_flag
from mimic.models import ImageAutoEncoder
from mimic.models import LSTMBase
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import AugedLSTM
from mimic.models import DenseBase
from mimic.models import DenseProp
from mimic.models import BiasedDenseProp
from mimic.models import DeprecatedDenseProp
from mimic.compat import is_compatible
from abc import ABC, abstractmethod

FBPropTypes = Union[LSTM, AugedLSTM, DenseProp, DeprecatedDenseProp]
FFPropTypes = Union[BiasedLSTM, BiasedDenseProp]
PropTypes = Union[FBPropTypes, FFPropTypes]

StateT = TypeVar('StateT') # TODO maybe this is unncessarly
FBPropT = TypeVar('FBPropT', bound=FBPropTypes)
FFPropT = TypeVar('FFPropT', bound=FFPropTypes)
PropT = TypeVar('PropT', bound=PropTypes)

class AbstractPredictor(ABC, Generic[StateT, PropT]):
    propagator: PropT
    states: List[torch.Tensor]
    def __init__(self, propagator: PropT):
        assert propagator.has_feature_info(), \
                'to use predictor you must set FeatureInfo to propagator model. Probably you may want to construct Config from Config.from_finfo function'
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
        return [self._strip_if_necessary(e) for e in raw_preds]

    def _is_with_aug(self): 
        config = self.propagator.config
        return (config.n_aug_feature != 0) and (config.n_cmd_feature != 0)

    def _attatch_if_ncecessary(self, vec: torch.Tensor) -> torch.Tensor:
        return self._attach_flag_if_necessary(self._attach_aug_if_necessary(vec))

    def _strip_if_necessary(self, vec: torch.Tensor) -> torch.Tensor:
        return self._strip_aug_if_necessary(self._strip_flag_if_necessary(vec))

    def _attach_aug_if_necessary(self, vec_original: torch.Tensor) -> torch.Tensor:
        if not self._is_with_aug():
            return vec_original

        config = self.propagator.config
        vec = vec_original[-config.n_cmd_feature:].unsqueeze(dim=0) # type: ignore

        propagator: AugedLSTM = self.propagator # type: ignore
        robot_spec = propagator.config.robot_spec
        fksolver = robot_spec.create_fksolver()

        pose = fksolver(vec)
        pose_torch = torch.from_numpy(pose).float().squeeze()
        vec = torch.hstack((vec_original, pose_torch))
        return vec

    def _strip_aug_if_necessary(self, vec: torch.Tensor) -> torch.Tensor:
        if not self._is_with_aug():
            return vec
        config = self.propagator.config
        return vec[:-config.n_aug_feature] # type: ignore

    def _is_with_flag(self):
        if isinstance(self.propagator, DeprecatedDenseProp): return False
        return isinstance(self.propagator, (LSTMBase, DenseBase))

    def _attach_flag_if_necessary(self, vec: torch.Tensor) -> torch.Tensor:
        if self._is_with_flag():
            flag = torch.tensor([_continue_flag])
            return torch.cat((vec, flag))
        return vec

    def _strip_flag_if_necessary(self, vec: torch.Tensor) -> torch.Tensor: 
        if self._is_with_flag():
            return vec[:-1]
        return vec

    def _force_continue_flag_if_necessary(self, vec: torch.Tensor) -> None: 
        if self._is_with_flag():
            vec[-1] = _continue_flag

    def _feed(self, state: torch.Tensor) -> None:
        state_maybe_with_flag = self._attatch_if_ncecessary(state)
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
        self._feed(torch.cat((img_feature, cmd_torch)))

    def predict(self, n_horizon: int, with_feeds: bool=False) -> List[MaybeNoneImageCommandPair]:
        preds = self._predict(n_horizon, with_feeds)
        cmd_list = [e.detach().numpy() for e in preds]
        return [(None, cmd) for cmd in cmd_list]

    # overwrite
    def _predict(self, n_horizon: int, with_feeds: bool=False) -> List[torch.Tensor]:
        feeds = copy.deepcopy(self.states) # cat with (img, cmd) which is different of super class's _predict
        preds: List[torch.Tensor] = [] # (cmd,) only
        assert self.img_torch_one_shot is not None
        for _ in range(n_horizon):
            states = torch.stack(feeds)
            tmp = self.propagator(torch.unsqueeze(states, 0))
            out = torch.squeeze(tmp, dim=0)[-1].detach().clone()
            self._force_continue_flag_if_necessary(out)
            feeds.append(torch.cat((self.img_torch_one_shot, out)))
            preds.append(out)
        if with_feeds:
            raw_preds = [s[self.auto_encoder.n_bottleneck:] for s in feeds]
        else:
            raw_preds = preds
        return [self._strip_if_necessary(e) for e in raw_preds]

def get_model_specific_state_slice(autoencoder: ImageAutoEncoder, propagator: PropTypes) -> slice:
    idx_start: Optional[int] = autoencoder.n_bottleneck
    idx_end = None
    if isinstance(propagator, (BiasedDenseProp, LSTMBase)):
        idx_end = -1
    return slice(idx_start, idx_end)

def create_predictor(autoencoder: ImageAutoEncoder, propagator: PropTypes) -> Union[ImageCommandPredictor, FFImageCommandPredictor]:
    if isinstance(propagator, (BiasedLSTM, BiasedDenseProp)):
        return FFImageCommandPredictor(propagator, autoencoder) # type: ignore
    else:
        return ImageCommandPredictor(propagator, autoencoder) # type: ignore
    raise RuntimeError

def evaluate_command_prediction_error(
        autoencoder: ImageAutoEncoder, 
        propagator: PropTypes, 
        chunk: ImageCommandDataChunk) -> float: 

    mse_list = []
    for seqs in chunk.seqs_list:
        predictor = create_predictor(autoencoder, propagator)
        cmd_pred_lst: List[np.ndarray] = []

        img_seq = None
        cmd_seq = None
        for seq in seqs:
            if isinstance(seq, ImageDataSequence): img_seq = seq
            if isinstance(seq, CommandDataSequence): cmd_seq = seq
        assert (img_seq is not None) and (cmd_seq is not None)

        for img, cmd in zip(img_seq.data[:-1], cmd_seq.data[:-1]):
            predictor.feed((img, cmd))
            pred_imgs, pred_cmds = zip(*predictor.predict(1))
            pred_cmd: np.ndarray = pred_cmds[0] # type: ignore
            cmd_pred_lst.append(pred_cmd)
        cmd_pred_seq = np.array(cmd_pred_lst)
        mse = ((cmd_pred_seq - cmd_seq.data[1:])**2).mean(axis=0).mean()
        mse_list.append(mse)

    return np.mean(mse_list) 

def evaluate_command_prediction_error_old(autoencoder: ImageAutoEncoder, propagator: PropTypes, 
        dataset: _DatasetFromChunk, batch_size: Optional[int] = None) -> float:
    # NOTE the result of this function will be sometimes largely different from the newer version.
    # This is because in this old version, the flag value is estimated by the predictor, while 
    # in the newer version, the flag value is forced to 1 (continue).

    assert is_compatible(propagator, dataset)
    #TODO check if dataset is compatible with propagator model
    slicer = get_model_specific_state_slice(autoencoder, propagator)

    if batch_size is None:
        batch_size = len(dataset)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    propagator.eval()
    loss_list = []
    for samples in loader:
        # TODO ? move to device?
        loss_dict = propagator.loss(samples, state_slicer=slicer, reduction='mean')
        loss_list.append(float(loss_dict['prediction'].item()))
    loss_mean = np.mean(loss_list)
    return loss_mean

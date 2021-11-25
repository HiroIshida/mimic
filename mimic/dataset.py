from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
from typing import Tuple
from typing import Type
from typing import Generic
from typing import TypeVar
import math
import numpy as np
import tinyfk

import torch
from torch.functional import Tensor
from torch.utils.data import Dataset

from mimic.robot import RobotSpecBase
from mimic.datatype import AbstractDataChunk
from mimic.datatype import ImageCommandDataChunk
from mimic.datatype import AugedImageCommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.robot import RobotSpecBase

from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

_continue_flag = 0.0
_end_flag = 1.0

ChunkT = TypeVar('ChunkT', bound=AbstractDataChunk)
DatasetT = TypeVar('DatasetT', bound='_DatasetFromChunk')
class _DatasetFromChunk(Dataset, Generic[ChunkT]):
    @classmethod
    def from_chunk(cls: Type[DatasetT], chunk: ChunkT) -> DatasetT: ...
    def __len__(self) -> int: ...

def compute_covariance_matrix(seqs_list: List[torch.Tensor]):
    diffs: List[torch.Tensor] = []
    for seqs in seqs_list:
        x_pre = seqs[:-1, :]
        x_post = seqs[1:, :]
        diff = x_post - x_pre
        diffs.append(diff)
    diffs_cat = torch.cat(diffs, dim=0)
    cov = torch.cov(diffs_cat.T)
    return cov

def augment_data(seqs_list: List[torch.Tensor], n_data_aug=10, cov_scale=0.3):
    if n_data_aug < 1:
        logger.info("because n_data_aug < 1, skip data augmentation process..")
        return seqs_list
    logger.info("augment data with parmas: n_data_aug {0}, cov_scale {1}".format(
        n_data_aug, cov_scale))
    cov = compute_covariance_matrix(seqs_list) * cov_scale ** 2
    cov_dim = cov.shape[0]
    walks_new = []
    for walk in seqs_list:
        n_seq, n_dim = walk.shape
        assert cov_dim == n_dim
        for _ in range(n_data_aug):
            rand_aug = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=cov, size=n_seq)
            assert rand_aug.shape == walk.shape
            walks_new.append(walk + torch.from_numpy(rand_aug).float())
    return walks_new

class ReconstructionDataset(_DatasetFromChunk[ImageDataChunk]):
    data: torch.Tensor
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_chunk(cls, chunk: ImageDataChunk) -> 'ReconstructionDataset':
        assert (not chunk.has_encoder)
        featureseq_list = chunk.to_featureseq_list()
        n_seq, n_channel, n_pixel1, n_pixel2 = featureseq_list[0].shape
        tmp = torch.cat(featureseq_list, dim=0)
        data = torch.reshape(tmp, (-1, n_channel, n_pixel1, n_pixel2))
        return ReconstructionDataset(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

def attach_flag_info(seq_list: List[torch.Tensor]) -> List[torch.Tensor]:
    n_state = len(seq_list[0][0])
    n_max = max([len(seq) for seq in seq_list])

    for i in range(len(seq_list)):
        seq = seq_list[i]
        n_seq = len(seq)
        n_padding = n_max - n_seq
        tensor_flags = torch.cat((torch.ones(n_seq) * _continue_flag, torch.ones(n_padding) * _end_flag))
        tensor_concat = seq[-1].repeat((n_padding, 1))
        tmp = torch.cat((seq, tensor_concat), dim=0)
        seq_list[i] = torch.cat((tmp, torch.unsqueeze(tensor_flags, 1)), dim=1)
    return seq_list

class AutoRegressiveDataset(_DatasetFromChunk):
    """
    Always come with end-of-epoch flags
    """
    data: List[torch.Tensor]
    def __init__(self, featureseq_list: List[torch.Tensor]):
        seq_list = deepcopy(featureseq_list)
        self.data = attach_flag_info(seq_list)

    @classmethod
    def from_chunk(cls, chunk: AbstractDataChunk, n_data_aug: int=0, cov_scale: float=0.3) -> 'AutoRegressiveDataset':
        if isinstance(chunk, ImageDataChunk) or isinstance(chunk, ImageCommandDataChunk):
            assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        featureseq_list_new = augment_data(featureseq_list, n_data_aug, cov_scale)
        return AutoRegressiveDataset(featureseq_list_new)

    @property
    def n_state(self) -> int: return self.data[0].shape[1]

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_input = self.data[idx][:-1]
        sample_output = self.data[idx][1:]
        return sample_input, sample_output

class AugedAutoRegressiveDataset(_DatasetFromChunk):
    data: List[torch.Tensor]
    n_aug: int
    robot_spec: RobotSpecBase
    def __init__(self, featureseq_list: List[torch.Tensor], n_aug: int, robot_sepec: RobotSpecBase):
        seq_list = deepcopy(featureseq_list)
        self.data = attach_flag_info(seq_list)
        self.n_aug = n_aug
        self.robot_spec = robot_sepec

    @classmethod
    def from_chunk(cls, chunk: AugedImageCommandDataChunk, n_data_aug: int=0, cov_scale: float=0.3) -> 'AugedAutoRegressiveDataset':
        assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        featureseq_list_new = augment_data(featureseq_list, n_data_aug, cov_scale)
        return cls(featureseq_list_new, chunk.n_aug, chunk.robot_spec)

    @property
    def n_state(self) -> int: return self.data[0].shape[1] - self.n_aug

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_input = self.data[idx][:-1]
        sample_output = self.data[idx][1:]
        return sample_input, sample_output

class BiasedAutoRegressiveDataset(_DatasetFromChunk[ImageCommandDataChunk]):
    data: List[torch.Tensor]
    n_bias: int
    def __init__(self, featureseq_list: List[torch.Tensor], n_bias: int):
        self.n_bias = n_bias
        seq_list = deepcopy(featureseq_list)
        self.data = attach_flag_info(seq_list)

    @classmethod
    def from_chunk(cls, chunk: ImageCommandDataChunk, n_data_aug: int=0, cov_scale: float=0.3) -> 'BiasedAutoRegressiveDataset':
        assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        featureseq_list_new = augment_data(featureseq_list, n_data_aug, cov_scale)
        return BiasedAutoRegressiveDataset(featureseq_list_new, chunk.n_encoder_output())

    @property
    def n_state(self) -> int: return self.data[0].shape[1] - self.n_bias
    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_input = self.data[idx][:-1]
        sample_output = self.data[idx][1:, self.n_bias:]
        return sample_input, sample_output

class FirstOrderARDataset(_DatasetFromChunk):
    n_state: int
    data_pre: torch.Tensor
    data_post: torch.Tensor
    def __init__(self, featureseq_list: List[torch.Tensor]):
        seq_list = deepcopy(featureseq_list)
        pre_list, post_list = [], []
        for seq in seq_list:
            pre, post = seq[:-1], seq[1:]
            pre_list.append(pre)
            post_list.append(post)
        self.data_pre = torch.cat(pre_list, dim=0)
        self.data_post = torch.cat(post_list, dim=0)
        self.n_state = len(self.data_pre[0])

    @classmethod
    def from_chunk(cls, chunk: AbstractDataChunk) -> 'FirstOrderARDataset':
        if isinstance(chunk, (ImageDataChunk, ImageCommandDataChunk)):
            assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        return FirstOrderARDataset(featureseq_list)

    def __len__(self) -> int: return len(self.data_pre)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        return (self.data_pre[idx], self.data_post[idx])

class BiasedFirstOrderARDataset(_DatasetFromChunk[ImageCommandDataChunk]):
    n_state: int
    n_bias: int
    biases: torch.Tensor # 2-dim tensor
    data_pre: torch.Tensor # 2-dim tensor
    data_post: torch.Tensor # 2-dim tensor
    def __init__(self, featureseq_list: List[torch.Tensor], n_encoder_output):
        seq_list = deepcopy(featureseq_list)
        pre_list, post_list, bias_list = [], [], []
        for seq in seq_list:
            n_seq, n_whole = seq.shape
            bias_image_feature_idx = 0
            bias = seq[bias_image_feature_idx, :n_encoder_output]
            bias_list.append(bias.unsqueeze(dim=0).expand((n_seq-1, -1)))

            seq_state = seq[:, n_encoder_output:]
            pre, post = seq_state[:-1], seq_state[1:]
            pre_list.append(pre)
            post_list.append(post)

        self.data_pre = torch.cat(pre_list, dim=0)
        self.data_post = torch.cat(post_list, dim=0)
        self.biases = torch.cat(bias_list, dim=0)
        self.n_bias = n_encoder_output
        self.n_state = len(self.data_pre[0] - n_encoder_output)

    @classmethod
    def from_chunk(cls, chunk: ImageCommandDataChunk) -> 'BiasedFirstOrderARDataset':
        assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        return BiasedFirstOrderARDataset(featureseq_list, chunk.n_encoder_output())

    def __len__(self) -> int: return len(self.data_pre)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        return (self.data_pre[idx], self.data_post[idx], self.biases[idx])

@dataclass
class KinematicsDataset(Dataset):
    data_input: torch.Tensor
    data_output: torch.Tensor
    robot_spec: RobotSpecBase
    n_joints: int
    def __init__(self, data_input, data_output, robot_spec: RobotSpecBase):
        self.data_input = data_input
        self.data_output = data_output
        self.robot_spec = robot_spec

    @classmethod
    def from_urdf(cls, robot_spec: RobotSpecBase, n_sample: Optional[int]=None):
        points = robot_spec.sample_from_cspace(n_sample)
        fksolver = robot_spec.create_fksolver()
        coords = fksolver(points)

        data_input = torch.from_numpy(points).float()
        data_output = torch.from_numpy(coords).float()
        return cls(data_input, data_output, robot_spec)

    def __len__(self) -> int: return len(self.data_input)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        return (self.data_input[idx], self.data_output[idx])

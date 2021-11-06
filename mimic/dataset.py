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

from mimic.datatype import AbstractDataChunk
from mimic.datatype import ImageCommandDataChunk
from mimic.datatype import AugedImageCommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.robot import RobotSpecBase

from dataclasses import dataclass

_continue_flag = 0.0
_end_flag = 1.0
_val_padding = 0.0

ChunkT = TypeVar('ChunkT', bound=AbstractDataChunk)
DatasetT = TypeVar('DatasetT', bound='_DatasetFromChunk')
class _DatasetFromChunk(Dataset, Generic[ChunkT]):
    @classmethod
    def from_chunk(cls: Type[DatasetT], chunk: ChunkT) -> DatasetT: ...
    def __len__(self) -> int: ...

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
        tensor_concat = torch.ones(n_padding, n_state) * _val_padding
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
    def from_chunk(cls, chunk: AbstractDataChunk) -> 'AutoRegressiveDataset':
        if isinstance(chunk, ImageDataChunk) or isinstance(chunk, ImageCommandDataChunk):
            assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        return AutoRegressiveDataset(featureseq_list)

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
    def from_chunk(cls, chunk: AugedImageCommandDataChunk) -> 'AugedAutoRegressiveDataset':
        assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        return cls(featureseq_list, chunk.n_aug, chunk.robot_spec)

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
    def from_chunk(cls, chunk: ImageCommandDataChunk) -> 'BiasedAutoRegressiveDataset':
        assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        return BiasedAutoRegressiveDataset(featureseq_list, chunk.n_encoder_output())

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
class KinematicsMetaData:
    joint_names: List[str] 
    link_names: List[str]

    @property
    def input_dim(self): return len(self.joint_names)
    @property
    def output_dim(self): return len(self.link_names) * (3 + 3) # trans + rot

class KinematicsDataset(Dataset):
    data_input: torch.Tensor
    data_output: torch.Tensor
    meta_data: KinematicsMetaData
    n_joints: int
    def __init__(self, data_input, data_output, meta_data):
        self.data_input = data_input
        self.data_output = data_output
        self.meta_data = meta_data

    @classmethod
    def from_urdf(cls, path_to_urdf: str, joint_names: List[str], link_names: List[str], n_sample: Optional[int]=None):
        n_joint = len(joint_names)
        if n_sample is None:
            n_sample = 8 ** n_joint
        kin_solver = tinyfk.RobotModel(path_to_urdf)
        joint_ids = kin_solver.get_joint_ids(joint_names)
        link_ids = kin_solver.get_link_ids(link_names)
        joint_limits = kin_solver.get_joint_limits(joint_ids)
        for i in range(len(joint_limits)):
            if joint_limits[i][0] == None:
                joint_limits[i][0] = -math.pi * 1.5
                joint_limits[i][1] = math.pi * 1.5
        lowers = np.array([limit[0] for limit in joint_limits])
        uppers = np.array([limit[1] for limit in joint_limits])
        sample_points = np.random.random((n_sample, n_joint)) * (uppers - lowers) + lowers

        coords, _ = kin_solver.solve_forward_kinematics(sample_points, link_ids, joint_ids, with_rot=True)
        coords = coords.reshape((-1, len(link_names) * 6))

        meta_data = KinematicsMetaData(joint_names, link_names)

        data_input = torch.from_numpy(sample_points).float()
        data_output = torch.from_numpy(coords).float()

        return cls(data_input, data_output, meta_data)

    def __len__(self) -> int: return len(self.data_input)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: 
        return (self.data_input[idx], self.data_output[idx])

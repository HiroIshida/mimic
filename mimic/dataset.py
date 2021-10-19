from copy import deepcopy
from typing import List
from typing import Tuple

import torch
from torch.functional import Tensor
from torch.utils.data import Dataset

from mimic.datatype import AbstractDataChunk, ImageCommandDataChunk
from mimic.datatype import ImageDataChunk

class ReconstructionDataset(Dataset):
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

def attach_flag_info(
        seq_list: List[torch.Tensor], 
        val_padding: float,
        continue_flag: float, 
        end_flag: float) -> List[torch.Tensor]:
    n_state = len(seq_list[0][0])
    n_max = max([len(seq) for seq in seq_list])

    for i in range(len(seq_list)):
        seq = seq_list[i]
        n_seq = len(seq)
        n_padding = n_max - n_seq
        tensor_flags = torch.cat((torch.ones(n_seq) * continue_flag, torch.ones(n_padding) * end_flag))
        tensor_concat = torch.ones(n_padding, n_state) * val_padding
        tmp = torch.cat((seq, tensor_concat), dim=0)
        seq_list[i] = torch.cat((tmp, torch.unsqueeze(tensor_flags, 1)), dim=1)
    return seq_list

class AutoRegressiveDataset(Dataset):
    """
    Always come with end-of-epoch flags
    """
    data: List[torch.Tensor]
    val_padding: float = 0.0
    continue_flag: float = 0.0
    end_flag: float = 1.0
    def __init__(self, featureseq_list: List[torch.Tensor], val_padding:float =0.0):
        seq_list = deepcopy(featureseq_list)
        self.data = attach_flag_info(seq_list, self.val_padding, self.continue_flag, self.end_flag)

    @classmethod
    def from_chunk(cls, chunk: AbstractDataChunk) -> 'AutoRegressiveDataset':
        if isinstance(chunk, ImageDataChunk):
            assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        return AutoRegressiveDataset(featureseq_list)

    @property
    def n_state(self) -> int: return self.data[0].shape[1]

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor: return self.data[idx]

class FirstOrderARDataset(Dataset):
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

class BiasedFirstOrderARDataset(Dataset):
    n_state: int
    n_bias: int
    bias_list: List[torch.Tensor] # list of 1-dim tensor
    data_pre: torch.Tensor # 2-dim tensor
    data_post: torch.Tensor # 2-dim tensor
    def __init__(self, featureseq_list: List[torch.Tensor], n_encoder_output):
        seq_list = deepcopy(featureseq_list)
        pre_list, post_list = [], []
        bias_list = []
        for seq in seq_list:
            n_seq, n_whole = seq.shape
            bias_image_feature_idx = 0
            bias = seq[bias_image_feature_idx, :n_encoder_output]
            assert bias.ndim == 1
            bias_list.append(bias)

            seq_state = seq[:, n_encoder_output:]
            pre, post = seq_state[:-1], seq_state[1:]
            pre_list.append(pre)
            post_list.append(post)

        self.data_pre = torch.cat(pre_list, dim=0)
        self.data_post = torch.cat(post_list, dim=0)
        self.bias_list = bias_list
        self.n_bias = n_encoder_output
        self.n_state = len(self.data_pre[0] - n_encoder_output)

    @classmethod
    def from_chunk(cls, chunk: ImageCommandDataChunk) -> 'BiasedFirstOrderARDataset':
        assert chunk.has_encoder
        featureseq_list = chunk.to_featureseq_list()
        return BiasedFirstOrderARDataset(featureseq_list, chunk.n_encoder_output())

    def __len__(self) -> int: return len(self.data_pre)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        return (self.data_pre[idx], self.data_post[idx], self.bias_list[idx])

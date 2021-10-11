from copy import deepcopy
from typing import List

import torch
from torch.functional import Tensor
from torch.utils.data import Dataset

from mimic.datatype import AbstractDataChunk
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
        tmp = torch.stack(featureseq_list)
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
    def n_state(self) -> int:
        return self.data[0].shape[1]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

from copy import deepcopy
from typing import List

import torch
from torch.utils.data import Dataset

from mimic.datatype import AbstractDataChunk
from mimic.datatype import ImageDataChunk

class ReconstructionDataset(Dataset):
    data: torch.Tensor
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_chunk(cls, chunk: ImageDataChunk) -> 'ReconstructionDataset':
        assert (not chunk.is_with_encode)
        featureseq_list = chunk.to_featureseq_list()
        n_seq, n_channel, n_pixel1, n_pixel2 = featureseq_list[0].shape
        tmp = torch.stack(featureseq_list)
        data = torch.reshape(tmp, (-1, n_channel, n_pixel1, n_pixel2))
        return ReconstructionDataset(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

class AutoRegressiveDataset(Dataset):
    data: List[torch.Tensor]
    def __init__(self, featureseq_list_: List[torch.Tensor], val_padding:float =1.0):
        featureseq_list = deepcopy(featureseq_list_)
        n_state = len(featureseq_list[0][0])
        n_max = max([len(seq) for seq in featureseq_list])

        for i in range(len(featureseq_list)):
            seq = featureseq_list[i]
            n_seq = len(seq)
            n_padding = n_max - n_seq
            tensor_add = torch.ones(n_padding, n_state) * val_padding
            featureseq_list[i] = torch.cat((seq, tensor_add), dim=0)
        self.data = featureseq_list

    @classmethod
    def from_chunk(cls, chunk: AbstractDataChunk) -> 'AutoRegressiveDataset':
        if isinstance(chunk, ImageDataChunk):
            assert chunk.is_with_encode
        featureseq_list = chunk.to_featureseq_list()
        return AutoRegressiveDataset(featureseq_list)


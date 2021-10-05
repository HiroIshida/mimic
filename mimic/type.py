from abc import ABC, abstractmethod
import torch
import numpy as np
import numpy.typing as npt
from typing import List, Dict

from torch.functional import Tensor

class DataSequence(ABC):
    data : npt.ArrayLike
    def __init__(self, data):
        self.data = data

    @abstractmethod
    def to_featureseq(self) -> torch.Tensor: ...

class CommandDataSequence(DataSequence):
    def to_featureseq(self):
        return torch.from_numpy(self.data)

class DataChunk:
    n_intact : int
    keys : List[type]
    seqdict_list : List[Dict[type, DataSequence]]

    def __init__(self):
        self.seqdict_list = []

    def push_epoch(self, seqs :List[DataSequence]):
        seqdict : Dict[type, DataSequence] = {}
        for seq in seqs:
            datatype = type(seq)
            seqdict[datatype] = seq

    def to_featureseq_list(self) -> List[torch.Tensor]:
        # a default converter method
        seqtorch_list = []
        for seqdict in self.seqdict_list:
            seqtorch = torch.cat([seqdict[key].to_featureseq() for key in self.keys])
            seqtorch_list.append(seqtorch)
        return seqtorch_list

from abc import ABC, abstractmethod
import torch
import numpy as np
import numpy.typing as npt
from typing import List, Dict

from torch.functional import Tensor

class AbstractDataSequence(ABC):
    data : npt.ArrayLike
    def __init__(self, data :npt.ArrayLike):
        self.data = data

    @abstractmethod
    def to_featureseq(self) -> torch.Tensor: ...

class CommandDataSequence(AbstractDataSequence):
    def to_featureseq(self):
        return torch.from_numpy(self.data)

class AbstractDataChunk(ABC):
    keys : List[type] = [] # override this
    n_intact : int
    seqdict_list : List[Dict[type, AbstractDataSequence]]

    def __init__(self):
        self.seqdict_list = []

    @abstractmethod
    def push_epoch(self, seqs) -> None: ...

    def _push_epoch(self, seqs :List[AbstractDataSequence]) -> None:
        assert set(self.keys) == set([type(e) for e in seqs])
        seqdict : Dict[type, AbstractDataSequence] = {}
        for seq in seqs:
            datatype = type(seq)
            seqdict[datatype] = seq
            self.seqdict_list.append(seqdict)

    def to_featureseq_list(self) -> List[torch.Tensor]:
        """
        each Tensor in return values is of the shape of (n_seq, n_feature) 
        """
        seqtorch_list = []
        for seqdict in self.seqdict_list:
            seqtorch = torch.cat([seqdict[key].to_featureseq() for key in self.keys])
            seqtorch_list.append(seqtorch)
        return seqtorch_list

class CommandDataChunk(AbstractDataChunk):
    keys = [CommandDataSequence]
    def push_epoch(self, seq: npt.ArrayLike) -> None:
        cmdseq = CommandDataSequence(seq)
        super()._push_epoch([cmdseq])

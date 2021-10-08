from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar

from torch.functional import Tensor

from mimic.file import dump_pickled_data
from mimic.file import load_pickled_data

class AbstractDataSequence(ABC):
    data : npt.ArrayLike
    def __init__(self, data :npt.ArrayLike):
        self.data = data

    @abstractmethod
    def to_featureseq(self) -> torch.Tensor: ...

ChunkT = TypeVar('ChunkT', bound='AbstractDataChunk')
class AbstractDataChunk(ABC):
    keys : List[type] = [] # override this
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

    @classmethod
    def load(cls: Type[ChunkT], project_name: str) -> ChunkT:
        return load_pickled_data(project_name, cls)

    def dump(self, project_name: str) -> None:
        dump_pickled_data(self, project_name, self.__class__)

class CommandDataSequence(AbstractDataSequence):
    def to_featureseq(self):
        return torch.from_numpy(self.data).float()

class CommandDataChunk(AbstractDataChunk):
    keys = [CommandDataSequence]
    def push_epoch(self, seq: npt.ArrayLike) -> None:
        cmdseq = CommandDataSequence(seq)
        super()._push_epoch([cmdseq])

class ImageDataSequence(AbstractDataSequence):
    encoder : Optional[nn.Module] = None
    def __init__(self, data: npt.ArrayLike, encoder : Optional[nn.Module] = None):
        super().__init__(data)
        self.encoder = encoder

    def to_featureseq(self) -> torch.Tensor:
        data_torch = torch.from_numpy(self.data).float()
        out = self.encoder(data_torch).detach() if self.encoder else data_torch
        return out

class ImageDataChunk(AbstractDataChunk):
    keys = [ImageDataSequence]
    encoder : Optional[nn.Module] = None
    def __init__(self, encoder: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder

    def push_epoch(self, seq: npt.ArrayLike) -> None:
        # TODO check if pil image type
        imgseq = ImageDataSequence(seq, self.encoder)
        super()._push_epoch([imgseq])

    @property
    def is_with_encode(self):
        return (self.encoder != None)

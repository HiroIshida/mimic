from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision
import numpy as np
from typing import Dict
from typing import List
from typing import Optional
from typing import Generic
from typing import Type
from typing import TypeVar
from typing import Tuple
from typing import NewType

from torch.functional import Tensor

from mimic.file import dump_pickled_data
from mimic.file import load_pickled_data

SeqT = TypeVar('SeqT', bound='AbstractDataSequence')
class AbstractDataSequence(ABC):
    data : np.ndarray
    def __init__(self, data: np.ndarray):
        self.data = data

    @abstractmethod
    def to_featureseq(self) -> torch.Tensor: ...

    def get_segment(self: SeqT, slicer: slice) -> SeqT:
        return self.__class__(self.data[slicer])

# TODO is there neat way to define like NewType('SeqDict', Dict[Type[SeqT], SeqT]) ?
class SeqDict(Dict, Generic[SeqT]):
    def __getitem__(self, key: Type[SeqT]) -> SeqT:
        return super().__getitem__(key)

ChunkT = TypeVar('ChunkT', bound='AbstractDataChunk')
class AbstractDataChunk(ABC):
    keys : List[type]
    seqdict_list : List[SeqDict]

    def __init__(self, seqdict_list: Optional[List[SeqDict]]=None):
        if seqdict_list is None:
            seqdict_list = []
        self.seqdict_list = seqdict_list

    @abstractmethod
    def push_epoch(self, seqs) -> None: ...

    def _push_epoch(self, seqs :List[AbstractDataSequence]) -> None:
        assert set(self.keys) == set([type(e) for e in seqs])
        seqdict: SeqDict = SeqDict({})
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
            seqtorch = torch.cat([seqdict[key].to_featureseq() for key in self.keys], dim=1)
            seqtorch_list.append(seqtorch)
        return seqtorch_list

    @classmethod
    def load(cls: Type[ChunkT], project_name: str) -> ChunkT:
        return load_pickled_data(project_name, cls)

    def dump(self, project_name: str) -> None:
        dump_pickled_data(self, project_name)

    def __getitem__(self, index: int) -> SeqDict:
        return self.seqdict_list[index]

class CommandDataSequence(AbstractDataSequence):
    def to_featureseq(self):
        return torch.from_numpy(self.data).float()

class CommandDataChunk(AbstractDataChunk):
    keys = [CommandDataSequence]
    def push_epoch(self, seq: np.ndarray) -> None:
        cmdseq = CommandDataSequence(seq)
        super()._push_epoch([cmdseq])

class ImageDataSequence(AbstractDataSequence):
    # the complex encoder_holder is due to lack of pointer-equivalent in python
    # if in C, I would wirite nn::Module* encoder_ptr;
    encoder_holder : Dict[str, Optional[nn.Module]]
    def __init__(self, data: np.ndarray, encoder_holder: Dict):
        super().__init__(data)
        self.encoder_holder = encoder_holder

    def to_featureseq(self) -> torch.Tensor:
        tf = torchvision.transforms.ToTensor()
        img_list = [tf(img).float() for img in self.data]
        data_torch = torch.stack(img_list)
        encoder = self.encoder_holder['encoder']
        out = encoder(data_torch).detach().clone() if encoder else data_torch
        return out

class ImageDataChunk(AbstractDataChunk):
    keys = [ImageDataSequence]
    encoder_holder : Dict[str, Optional[nn.Module]] = {'encoder': None}
    def __init__(self, 
            encoder: Optional[nn.Module] = None, 
            seqdict_list: Optional[List[SeqDict]] = None):
        if seqdict_list is None:
            seqdict_list = []
        super().__init__(seqdict_list)
        self.encoder_holder = {'encoder': encoder}

    def push_epoch(self, seq: np.ndarray) -> None:
        imgseq = ImageDataSequence(seq, self.encoder_holder)
        super()._push_epoch([imgseq])

    def set_encoder(self, encoder: nn.Module):
        self.encoder_holder['encoder'] = encoder

    @property
    def has_encoder(self):
        return (self.encoder_holder['encoder'] != None)

    @classmethod
    def from_imgcmd_chunk(cls, chunk: 'ImageCommandDataChunk') -> 'ImageDataChunk':
        seqdict_list_new: List[SeqDict] = []
        for seqdict in chunk.seqdict_list:
            val = seqdict[ImageDataSequence]
            newdict: SeqDict = SeqDict({ImageDataSequence : val})
            seqdict_list_new.append(newdict)
        return ImageDataChunk(seqdict_list=seqdict_list_new)

class ImageCommandDataChunk(AbstractDataChunk):
    keys = [ImageDataSequence, CommandDataSequence]
    encoder_holder : Dict[str, Optional[nn.Module]] = {'encoder': None}
    def __init__(self, encoder: Optional[nn.Module] = None):
        super().__init__()
        self.encoder_holder = {'encoder': encoder}

    def push_epoch(self, imgcmd_seq: Tuple[np.ndarray, np.ndarray]) -> None:
        imgseq, cmdseq = imgcmd_seq
        assert imgseq.ndim == 4
        assert cmdseq.ndim == 2
        img_data_seq = ImageDataSequence(imgseq, self.encoder_holder)
        cmd_data_seq = CommandDataSequence(cmdseq)
        super()._push_epoch([img_data_seq, cmd_data_seq])

    def set_encoder(self, encoder: nn.Module):
        self.encoder_holder['encoder'] = encoder

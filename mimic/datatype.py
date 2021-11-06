from abc import ABC, abstractmethod
import copy
import torch
import torch.nn as nn
import torchvision
import numpy as np
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Generic
from typing import Type
from typing import TypeVar
from typing import Tuple
from typing import NewType

import tinyfk

from torch.functional import Tensor

from mimic.file import dump_pickled_data
from mimic.file import load_pickled_data
from mimic.robot import RobotSpecBase
from mimic.primitives import AbstractEncoder


SeqT = TypeVar('SeqT', bound='AbstractDataSequence')
class AbstractDataSequence(ABC):
    data : np.ndarray
    def __init__(self, data: np.ndarray):
        self.data = data

    @abstractmethod
    def to_featureseq(self) -> torch.Tensor: ...

    def get_segment(self: SeqT, slicer: Any) -> SeqT:
        # TODO(HiroIshida) too dirty. there must be a sane way...
        obj = copy.deepcopy(self)
        obj.data = self.data[slicer]
        return obj

DataT = TypeVar('DataT', bound=Tuple[AbstractDataSequence, ...])
ChunkT = TypeVar('ChunkT', bound='AbstractDataChunk')
class AbstractDataChunk(ABC, Generic[DataT]):
    seqs_list : List[DataT]

    def __init__(self, seqs_list: Optional[List[DataT]]=None):
        if seqs_list is None:
            seqs_list = []
        self.seqs_list = seqs_list

    @abstractmethod
    def push_epoch(self, seqs) -> None: ...

    def _push_epoch(self, seqs :DataT) -> None: 
        self.seqs_list.append(seqs)

    def to_featureseq_list(self) -> List[torch.Tensor]:
        """
        each Tensor in return values is of the shape of (n_seq, n_feature) 
        """
        seqtorch_list = []
        for seqs in self.seqs_list:
            seqtorch = torch.cat([e.to_featureseq() for e in seqs], dim=1)
            seqtorch_list.append(seqtorch)
        return seqtorch_list

    @classmethod
    def load(cls: Type[ChunkT], project_name: str) -> ChunkT:
        data_list = load_pickled_data(project_name, cls)
        assert len(data_list) == 1
        return data_list[0]

    def dump(self, project_name: str) -> None:
        obj = copy.deepcopy(self)
        dump_pickled_data(obj, project_name)

    def split(self: ChunkT, n_first) -> Tuple[ChunkT, ChunkT]:
        first = copy.copy(self)
        second = copy.copy(self)
        first.seqs_list = self.seqs_list[:n_first]
        second.seqs_list = self.seqs_list[n_first:]
        return first, second

    def __len__(self) -> int: return len(self.seqs_list)

    def __getitem__(self, index: int) -> DataT:
        return self.seqs_list[index]

class VectorDataSequence(AbstractDataSequence):
    def to_featureseq(self):
        return torch.from_numpy(self.data).float()

class CommandDataSequence(VectorDataSequence): ...
class AugDataSequence(VectorDataSequence): ...

_CommandDataSequence = Tuple[CommandDataSequence]
class CommandDataChunk(AbstractDataChunk[_CommandDataSequence]):
    def push_epoch(self, seq: np.ndarray) -> None:
        cmdseq = CommandDataSequence(seq)
        super()._push_epoch((cmdseq,))

class ImageDataSequence(AbstractDataSequence):
    # the complex encoder_holder is due to lack of pointer-equivalent in python
    # if in C, I would wirite nn::Module* encoder_ptr;
    encoder_holder : Dict[str, Optional[AbstractEncoder]]
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

class ImageDataChunkBase:
    encoder_holder : Dict[str, Optional[AbstractEncoder]]
    def __init__(self, encoder: Optional[AbstractEncoder]):
        self.encoder_holder = {'encoder': encoder}

    def set_encoder(self, encoder: Optional[AbstractEncoder]) -> None:
        self.encoder_holder['encoder'] = encoder

    @property
    def has_encoder(self) -> bool:
        return (self.encoder_holder['encoder'] != None)

    @typing.no_type_check 
    def n_encoder_output(self) -> Optional[int]:
        if self.encoder_holder['encoder']: # because mypy is dumb, cannot use has_encoder
            return self.encoder_holder['encoder'].n_output
        return None

_ImageDataSequence = Tuple[ImageDataSequence]
class ImageDataChunk(AbstractDataChunk[_ImageDataSequence], ImageDataChunkBase):
    def __init__(self, 
            encoder: Optional[AbstractEncoder] = None, 
            seqs_list: Optional[List[_ImageDataSequence]] = None):
        if seqs_list is None:
            seqs_list = []
        AbstractDataChunk.__init__(self, seqs_list)
        ImageDataChunkBase.__init__(self, encoder)

    def push_epoch(self, seq: np.ndarray) -> None:
        imgseq = ImageDataSequence(seq, self.encoder_holder)
        super()._push_epoch((imgseq,))

    @classmethod
    def from_imgcmd_chunk(cls, chunk: 'ImageCommandDataChunk') -> 'ImageDataChunk':
        seqs_list_new: List[_ImageDataSequence] = []
        for seqs in chunk.seqs_list:
            for seq in seqs:
                if isinstance(seq, ImageDataSequence):
                    seqs_list_new.append((seq,))
        return ImageDataChunk(seqs_list=seqs_list_new)

_ImageCommandDataSequence = Tuple[ImageDataSequence, CommandDataSequence]
class ImageCommandDataChunk(AbstractDataChunk[_ImageCommandDataSequence], ImageDataChunkBase):
    def __init__(self, encoder: Optional[AbstractEncoder] = None):
        super().__init__([]) # TODO enable optional seq input??
        ImageDataChunkBase.__init__(self, encoder)

    def push_epoch(self, imgcmd_seq: Tuple[np.ndarray, np.ndarray]) -> None:
        imgseq, cmdseq = imgcmd_seq
        assert imgseq.ndim == 4 and cmdseq.ndim == 2
        img_data_seq = ImageDataSequence(imgseq, self.encoder_holder)
        cmd_data_seq = CommandDataSequence(cmdseq)
        super()._push_epoch((img_data_seq, cmd_data_seq))

_AugedImageCommandDataSequence = Tuple[ImageDataSequence, CommandDataSequence, AugDataSequence]
class AugedImageCommandDataChunk(AbstractDataChunk[_AugedImageCommandDataSequence], ImageDataChunkBase):
    def __init__(self, encoder: Optional[AbstractEncoder] = None):
        super().__init__([]) # TODO enable optional seq input??
        ImageDataChunkBase.__init__(self, encoder)

    def push_epoch(self, auged_imgcmd_seq: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """
        auged_imgcmd_seq with (img, cmd, aug) order
        """
        imgseq, cmdseq, augseq = auged_imgcmd_seq
        assert imgseq.ndim == 4 and cmdseq.ndim == 2
        img_data_seq = ImageDataSequence(imgseq, self.encoder_holder)
        cmd_data_seq = CommandDataSequence(cmdseq)
        aug_data_seq = AugDataSequence(augseq)
        super()._push_epoch((img_data_seq, cmd_data_seq, aug_data_seq))

    @classmethod
    def from_imgcmd_chunk(cls, chunk_other: ImageCommandDataChunk, 
            robot_spec: RobotSpecBase) -> 'AugedImageCommandDataChunk':

        img_seq, cmd_seq = chunk_other.seqs_list[0]
        _, n_dof = cmd_seq.data.shape
        assert n_dof == len(robot_spec.joint_names)

        kin_solver = tinyfk.RobotModel(robot_spec.urdf_path)
        joint_ids = kin_solver.get_joint_ids(robot_spec.joint_names)
        link_ids = kin_solver.get_link_ids(robot_spec.featured_link_names)

        obj = cls(chunk_other.encoder_holder['encoder'])
        for img_seq, cmd_seq in chunk_other.seqs_list:
            angle_vectors = cmd_seq.data
            coords, _ = kin_solver.solve_forward_kinematics(angle_vectors, link_ids, joint_ids, with_rot=True)
            coords = coords.reshape((-1, len(robot_spec.featured_link_names) * 6))
            obj.push_epoch((img_seq.data, cmd_seq.data, coords))
        return obj

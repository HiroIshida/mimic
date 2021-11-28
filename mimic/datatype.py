from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
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

from torch.functional import Tensor
import albumentations as A

from mimic.file import dump_pickled_data
from mimic.file import load_pickled_data
from mimic.robot import RobotSpecBase
from mimic.primitives import AbstractEncoder

# TODO
# class is not flexible, meaning that when a user want to add another 
# property such as audio feature, user how to modiry the library.
# So, the class idearly should be a dict.
# However, for my own use class is sufficient, because I am the god.
@dataclass
class FeatureInfo:
    n_img_feature: int=0
    n_cmd_feature: int=0
    n_aug_feature: int=0

SeqT = TypeVar('SeqT', bound='AbstractDataSequence')
class AbstractDataSequence(ABC):
    data : np.ndarray
    def __init__(self, data: np.ndarray):
        self.data = data

    @abstractmethod
    def to_featureseq(self) -> torch.Tensor: ...

    @abstractmethod
    def edit_feature_info(self, fi: FeatureInfo) -> None: ...

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

    def get_feature_info(self) -> FeatureInfo:
        seqs = self.seqs_list[0]
        fi = FeatureInfo()
        for seq in seqs:
            seq.edit_feature_info(fi)
        return fi

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

class CommandDataSequence(VectorDataSequence):
    def edit_feature_info(self, fi: FeatureInfo):
        fi.n_cmd_feature = self.data.shape[1]

class AugDataSequence(VectorDataSequence):
    def edit_feature_info(self, fi: FeatureInfo):
        fi.n_aug_feature = self.data.shape[1]

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
        encoder = self.encoder_holder['encoder']
        with_feature_encoding = encoder is not None

        # TODO(HiroIshida) too ad-hoc impl
        if with_feature_encoding:
            # no aug
            np_data = self.data
        else:
            f_aug = A.Compose([A.GaussNoise(p=1), A.RGBShift(p=1)])
            auged_imgseq_list = []
            _n_image_data_aug = 9
            for _ in range(_n_image_data_aug):
                aug_seq = np.array([f_aug(image=img)['image'] for img in self.data])
                auged_imgseq_list.append(aug_seq)
                print(aug_seq.shape)
            auged_imgseq_list.append(self.data)
            np_data = np.concatenate(auged_imgseq_list, axis=0)

        tf = torchvision.transforms.ToTensor()
        img_list = [tf(img).float() for img in np_data]
        data_torch = torch.stack(img_list)
        if with_feature_encoding:
            out = encoder(data_torch).detach().clone() # type: ignore
        else:
            out = data_torch
        return out

    def edit_feature_info(self, fi: FeatureInfo):
        fi.n_img_feature = 0
        encoder = self.encoder_holder['encoder']
        if encoder is not None:
            fi.n_img_feature = encoder.n_output

class ImageDataChunkBase(AbstractDataChunk[DataT]):
    encoder_holder : Dict[str, Optional[AbstractEncoder]]
    def __init__(self, seqs_list: Optional[List[DataT]]=None, encoder: Optional[AbstractEncoder]=None):
        super().__init__(seqs_list)
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
class ImageDataChunk(ImageDataChunkBase[_ImageDataSequence]):

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
class ImageCommandDataChunk(ImageDataChunkBase[_ImageCommandDataSequence]):

    def push_epoch(self, imgcmd_seq: Tuple[np.ndarray, np.ndarray]) -> None:
        imgseq, cmdseq = imgcmd_seq
        assert imgseq.ndim == 4 and cmdseq.ndim == 2
        img_data_seq = ImageDataSequence(imgseq, self.encoder_holder)
        cmd_data_seq = CommandDataSequence(cmdseq)
        super()._push_epoch((img_data_seq, cmd_data_seq))

_AugedImageCommandDataSequence = Tuple[ImageDataSequence, CommandDataSequence, AugDataSequence]
class AugedImageCommandDataChunk(ImageDataChunkBase[_AugedImageCommandDataSequence]):
    robot_spec: RobotSpecBase
    def __init__(self, 
            robot_spec: RobotSpecBase, 
            seqs_list: Optional[List[_AugedImageCommandDataSequence]]=None, 
            encoder: Optional[AbstractEncoder] = None):
        ImageDataChunkBase.__init__(self, seqs_list=seqs_list, encoder=encoder)
        self.robot_spec = robot_spec

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

    @property
    def n_aug(self) -> int: 
        img_seq, cmd_seq, aug_seq = self.seqs_list[0]
        return aug_seq.data.shape[1]

    @classmethod
    def from_imgcmd_chunk(cls, chunk_other: ImageCommandDataChunk, 
            robot_spec: RobotSpecBase) -> 'AugedImageCommandDataChunk':
        img_seq, cmd_seq = chunk_other.seqs_list[0]
        _, n_dof = cmd_seq.data.shape
        assert n_dof == len(robot_spec.joint_names)

        fksolver = robot_spec.create_fksolver()

        obj = cls(robot_spec, encoder=chunk_other.encoder_holder['encoder'])
        for img_seq, cmd_seq in chunk_other.seqs_list:
            coords = fksolver(cmd_seq.data)
            obj.push_epoch((img_seq.data, cmd_seq.data, coords))
        return obj

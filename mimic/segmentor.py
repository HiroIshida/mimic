from functools import lru_cache
from copy import deepcopy
from cached_property import cached_property
from typing import List
from typing import TypeVar
import numpy as np
from mimic.datatype import AbstractDataChunk

ChunkT = TypeVar('ChunkT', bound=AbstractDataChunk)
class Segmentor:
    n_phase: int
    data: List[np.ndarray]
    def __init__(self, data: List[np.ndarray]):
        n_epoch = len(data)
        n_seq, n_phase = data[0].shape
        self.n_phase = n_phase
        self.data = data

    @lru_cache
    def to_labelseq_list(self) -> List[np.ndarray]:
        return [np.array([np.argmax(vec) for vec in seq]) for seq in self.data]

    @cached_property
    def is_ordered(self) -> bool:
        for labels in self.to_labelseq_list():
            for i in range(len(labels)-1):
                if labels[i] > labels[i+1]:
                    return False
        return True

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.to_labelseq_list()[idx]

    def __call__(self, chunk: ChunkT) -> List[ChunkT]:
        assert self.is_ordered, "currently support only ordered case"
        chunk_list = []
        for phase in range(self.n_phase):
            chunk_new = deepcopy(chunk)
            chunk_new.seqs_list = [] # clear old one
            for seqs, labelseq in zip(chunk.seqs_list, self.to_labelseq_list()):
                idxes = np.nonzero(labelseq == phase)[0]
                seqs_new = tuple(seq.get_segment(idxes) for seq in seqs)
                chunk_new.seqs_list.append(seqs_new)
            chunk_list.append(chunk_new)
        return chunk_list

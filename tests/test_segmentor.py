import pytest
import numpy as np
from mimic.segmentor import Segmentor
from mimic.datatype import CommandDataChunk

@pytest.fixture(scope="module")
def dummy_chunk():
    chunk = CommandDataChunk()
    n_seq = 9
    n_dim = 3
    for _ in range(2):
        chunk.push_epoch(np.random.randn(n_seq, n_dim))
    return chunk

def onehot(idx):
    a = np.zeros(3)
    a[idx] = 1.0
    return a

@pytest.fixture(scope="module")
def segmentation_raw_ordered():
    seq1 = [onehot(0) for _ in range(3)] + [onehot(1) for _ in range(3)] + [onehot(2) for _ in range(3)]
    seq2 = [onehot(0) for _ in range(3)] + [onehot(1) for _ in range(6)]
    return [np.array(seq1), np.array(seq2)]

@pytest.fixture(scope="module")
def segmentation_raw_unordered():
    seq1 = [onehot(0) for _ in range(3)] + [onehot(1) for _ in range(3)] + [onehot(2) for _ in range(3)]
    seq2 = [onehot(0) for _ in range(3)] + [onehot(1) for _ in range(4)] + [onehot(0) for _ in range(3)]
    return [np.array(seq1), np.array(seq2)]

def test_ordered(segmentation_raw_ordered, segmentation_raw_unordered):
    seg1 = Segmentor(segmentation_raw_ordered)
    assert seg1.is_ordered

    seg2 = Segmentor(segmentation_raw_unordered)
    assert ~seg2.is_ordered

def test_segmentation_pipeline(dummy_chunk, segmentation_raw_ordered):
    chunk: CommandDataChunk = dummy_chunk
    seg = Segmentor(segmentation_raw_ordered)
    from typing import List
    chunk_list: List[CommandDataChunk] = seg.__call__(dummy_chunk)
    assert len(chunk_list)==seg.n_phase

    c1, c2, c3 = chunk_list
    assert len(c1.seqs_list[0][0].data) == 3
    assert len(c1.seqs_list[1][0].data) == 3
    assert len(c2.seqs_list[0][0].data) == 3
    assert len(c2.seqs_list[1][0].data) == 6
    assert len(c3.seqs_list[0][0].data) == 3
    assert len(c3.seqs_list[1][0].data) == 0

def test_dump_load(segmentation_raw_ordered):
    seg = Segmentor(segmentation_raw_ordered)
    seg.dump('test')
    seg_loaded = Segmentor.load('test')

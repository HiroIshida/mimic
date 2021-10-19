import torch
import pytest
import numpy as np
from mimic.segmentor import Segmentor
from mimic.datatype import CommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.models import ImageAutoEncoder

@pytest.fixture(scope="module")
def dummy_cmd_chunk():
    chunk = CommandDataChunk()
    n_seq = 9
    n_dim = 3
    for _ in range(2):
        chunk.push_epoch(np.random.randn(n_seq, n_dim))
    return chunk

@pytest.fixture(scope="module")
def dummy_img_chunk():
    # TODO duplication
    n_seq = 9
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    chunk = ImageDataChunk(encoder=ae.get_encoder())
    for i in range(2):
        imgseq = np.random.randn(n_seq, n_pixel, n_pixel, n_channel)
        chunk.push_epoch(imgseq)
    chunk.push_epoch(imgseq)
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
    assert seg1.is_ordered()

    seg2 = Segmentor(segmentation_raw_unordered)
    assert ~seg2.is_ordered()

def test_segmentation_pipeline(dummy_cmd_chunk, segmentation_raw_ordered):
    chunk: CommandDataChunk = dummy_cmd_chunk
    seg = Segmentor(segmentation_raw_ordered)
    chunk_list = seg.__call__(dummy_cmd_chunk)
    assert len(chunk_list)==seg.n_phase

    c1, c2, c3 = chunk_list
    assert len(c1.seqs_list[0][0].data) == 3
    assert len(c1.seqs_list[1][0].data) == 3
    assert len(c2.seqs_list[0][0].data) == 3
    assert len(c2.seqs_list[1][0].data) == 6
    assert len(c3.seqs_list[0][0].data) == 3
    assert len(c3.seqs_list[1][0].data) == 0

def test_segmentation_pipeline2(dummy_img_chunk, segmentation_raw_ordered):
    chunk: ImageDataChunk = dummy_img_chunk
    seg = Segmentor(segmentation_raw_ordered)
    chunk_list = seg.__call__(chunk)
    assert len(chunk_list)==seg.n_phase

def test_dump_load(segmentation_raw_ordered):
    seg = Segmentor(segmentation_raw_ordered)
    seg.dump('test')
    seg_loaded = Segmentor.load('test')

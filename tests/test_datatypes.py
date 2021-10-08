import hashlib
import pytest
import numpy as np
import torch

from mimic.datatype import CommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.models import ImageAutoEncoder

@pytest.fixture(scope='session')
def cmd_datachunk():
    chunk = CommandDataChunk()
    for i in range(10):
        seq = np.zeros((20, 7))
        chunk.push_epoch(seq)
    return chunk

@pytest.fixture(scope='session')
def image_datachunk():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    chunk = ImageDataChunk()
    for i in range(10):
        imgseq = np.random.randn(n_seq, n_channel, n_pixel, n_pixel)
        chunk.push_epoch(imgseq)
    return chunk

@pytest.fixture(scope='session')
def image_datachunk_with_encoder():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(16, torch.device('cpu'), image_shape=(n_channel, n_pixel, n_pixel))
    chunk = ImageDataChunk(encoder=ae.encoder)
    for i in range(9):
        imgseq = np.random.randn(n_seq, n_channel, n_pixel, n_pixel)
        chunk.push_epoch(imgseq)
    imgseq = np.random.randn(n_seq-2, n_channel, n_pixel, n_pixel) # to test autoregressive
    chunk.push_epoch(imgseq)
    return chunk

def test_dump_load(cmd_datachunk):
    cmd_datachunk.dump("test")
    chunk = CommandDataChunk.load("test")

def test_featureseq_list_generation_pipeline(cmd_datachunk):
    fslist = cmd_datachunk.to_featureseq_list()
    assert len(fslist) == 10
    assert list(fslist[0].size()) == [20, 7]

def test_image_featureseq_list_generateion_pipeline(image_datachunk, image_datachunk_with_encoder):
    fslist = image_datachunk_with_encoder.to_featureseq_list()
    assert len(fslist[0].size()) == 2
    assert list(fslist[0].size()) == [100, 16]

    fslist = image_datachunk.to_featureseq_list()
    assert len(fslist[0].size()) == 4
    assert list(fslist[0].size()) == [100, 3, 28, 28]

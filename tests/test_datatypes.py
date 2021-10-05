import pytest
import numpy as np
import torch

from mimic.datatype import CommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.models import ImageAutoEncoder

@pytest.fixture(scope='module')
def cmd_datachunk():
    chunk = CommandDataChunk()
    for i in range(10):
        seq = np.zeros((20, 7))
        chunk.push_epoch(seq)
    return chunk

def test_featureseq_list_generation_pipeline(cmd_datachunk):
    fslist = cmd_datachunk.to_featureseq_list()
    assert len(fslist) == 10
    assert list(fslist[0].size()) == [20, 7]

def test_image_featureseq_list_generateion_pipeline():
    n_batch = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(16, torch.device('cpu'), image_shape=(n_channel, n_pixel, n_pixel))

    chunk = ImageDataChunk(ae.encoder)
    for i in range(10):
        imgseq = np.random.randn(n_batch, n_channel, n_pixel, n_pixel)
        chunk.push_epoch(imgseq)
    fslist = chunk.to_featureseq_list()
    assert len(fslist[0].size()) == 2
    assert list(fslist[0].size()) == [100, ae.n_bottleneck]


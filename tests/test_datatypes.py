import hashlib
import pytest
import numpy as np
import torch
import copy

from mimic.datatype import CommandDataSequence
from mimic.datatype import CommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.datatype import ImageCommandDataChunk
from mimic.datatype import AugedImageCommandDataChunk
from mimic.models import ImageAutoEncoder
from mimic.robot import KukaSpec

def test_dataseq_slice():
    seq = CommandDataSequence(np.zeros((10, 3)))
    new_seq = seq.get_segment(slice(0, 3))
    assert new_seq.data.shape == (3, 3)

@pytest.fixture(scope='session')
def cmd_datachunk():
    chunk = CommandDataChunk()
    for i in range(10):
        seq = np.zeros((20, 7))
        chunk.push_epoch(seq)
    return chunk

def test_datasplit(cmd_datachunk):
    chunk: CommandDataChunk = cmd_datachunk
    chunk1, chunk2 = chunk.split(2)
    assert len(chunk1) == 2
    assert len(chunk2) == 8

_img_chunk_uneven_n = 2
@pytest.fixture(scope='session')
def image_datachunk():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    chunk = ImageDataChunk()
    for i in range(10):
        if i==9:
            n_seq = n_seq + _img_chunk_uneven_n # to test uneven dataset 
        imgseq = np.random.randint(256, size=(n_seq, n_pixel, n_pixel, n_channel), dtype=np.uint8)
        chunk.push_epoch(imgseq)
    assert not chunk.with_depth
    return chunk

def test_depthimage_datachunk():
    n_seq = 100
    n_channel = 4
    n_pixel = 28
    chunk = ImageDataChunk()
    for i in range(10):
        imgseq = np.random.randint(256, size=(n_seq, n_pixel, n_pixel, n_channel), dtype=np.uint8)
        chunk.push_epoch(imgseq)
    assert chunk.with_depth

    chunk_no_depth = chunk.to_depth_stripped()
    assert not chunk_no_depth.with_depth

@pytest.fixture(scope='session')
def image_datachunk_with_encoder():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    chunk = ImageDataChunk(encoder=ae.get_encoder())
    for i in range(9):
        imgseq = np.random.randint(256, size=(n_seq, n_pixel, n_pixel, n_channel), dtype=np.uint8)
        chunk.push_epoch(imgseq)
    imgseq = np.random.randint(256, size=(n_seq, n_pixel, n_pixel, n_channel), dtype=np.uint8)
    chunk.push_epoch(imgseq)

    fi = chunk.get_feature_info()
    assert fi.n_img_feature == 16
    assert fi.n_cmd_feature == 0
    assert fi.n_aug_feature == 0
    return chunk

@pytest.fixture(scope='session')
def image_command_datachunk_with_encoder():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    chunk = ImageCommandDataChunk(encoder=ae.get_encoder())
    for i in range(10):
        imgseq = np.random.randint(256, size=(n_seq, n_pixel, n_pixel, n_channel), dtype=np.uint8)
        cmdseq = np.random.randn(n_seq, 7)
        chunk.push_epoch((imgseq, cmdseq))

    fi = chunk.get_feature_info()
    assert fi.n_img_feature == 16
    assert fi.n_cmd_feature == 7
    assert fi.n_aug_feature == 0

    return chunk

def test_dump_load(cmd_datachunk):
    cmd_datachunk.dump("test")
    chunk = CommandDataChunk.load("test")

def test_featureseq_list_generation_pipeline(cmd_datachunk):
    fslist = cmd_datachunk.to_featureseq_list()
    assert len(fslist) == 10
    assert list(fslist[0].size()) == [20, 7]

def test_set_encoder(image_datachunk):
    chunk: ImageDataChunk = copy.deepcopy(image_datachunk)
    flist = chunk.to_featureseq_list()
    assert len(flist[0].shape) == 4

    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(torch.device('cpu'), 16, image_shape=(n_channel, n_pixel, n_pixel))
    assert not chunk.has_encoder
    chunk.set_encoder(ae.get_encoder())
    assert chunk.has_encoder
    flist = chunk.to_featureseq_list()
    assert len(flist[0].shape) == 2

def test_image_featureseq_list_generateion_pipeline(image_datachunk, image_datachunk_with_encoder):
    fslist = image_datachunk_with_encoder.to_featureseq_list()
    assert len(fslist[0].size()) == 2
    assert list(fslist[0].size()) == [100, 16]

    fslist = image_datachunk.to_featureseq_list()
    assert len(fslist[0].size()) == 4
    assert list(fslist[0].size()) == [100, 3, 28, 28]

def test_image_command_datachunk_with_encoder_pipeline(image_command_datachunk_with_encoder):
    fslist = image_command_datachunk_with_encoder.to_featureseq_list()
    assert list(fslist[0].size()) == [100, 16 + 7]

@pytest.fixture(scope='session')
def auged_image_command_datachunk(image_command_datachunk_with_encoder):
    chunk_other: ImageCommandDataChunk = image_command_datachunk_with_encoder
    chunk = AugedImageCommandDataChunk.from_imgcmd_chunk(chunk_other, KukaSpec())

    fi = chunk.get_feature_info()
    assert fi.n_img_feature == 16
    assert fi.n_cmd_feature == 7
    assert fi.n_aug_feature == KukaSpec().n_out

    return chunk

def test_auged_image_command_datachunk_pipeline(auged_image_command_datachunk):
    fslist = auged_image_command_datachunk.to_featureseq_list()
    assert list(fslist[0].size()) == [100, 16 + 7 + 6]

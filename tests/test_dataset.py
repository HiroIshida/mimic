import numpy as np
import torch
from mimic.datatype import CommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.dataset import ReconstructionDataset
from mimic.dataset import attach_flag_info
from mimic.dataset import AutoRegressiveDataset
import pytest

from test_datatypes import cmd_datachunk
from test_datatypes import image_datachunk
from test_datatypes import image_datachunk_with_encoder

def test_reconstruction_dataset_pipeline(image_datachunk):
    dataset = ReconstructionDataset.from_chunk(image_datachunk)
    assert list(dataset.data.shape) == [10 * 100, 3, 28, 28]
    assert len(dataset) == 10 * 100
    assert list(dataset[0].shape) == [3, 28, 28]

def test_attach_flag_info():
    seq1 = torch.randn((10, 3))
    seq2 = torch.randn((12, 3))
    seq3 = torch.randn((14, 3))
    seq_list = [seq1, seq2, seq3]
    val_padding = 0.0
    seq_list_with_flag = attach_flag_info(seq_list, val_padding, 0., 1.)

    assert seq_list_with_flag[0][9, 3] == 0.0
    assert seq_list_with_flag[0][10, 3] == 1.0
    assert seq_list_with_flag[1][11, 3] == 0.0
    assert seq_list_with_flag[1][12, 3] == 1.0
    assert seq_list_with_flag[2][13, 3] == 0.0

    assert seq_list_with_flag[0][10:, :3].sum() == 0.0
    assert seq_list_with_flag[0][12:, :3].sum() == 0.0

def test_autoregressive_dataset_pipeline1(image_datachunk_with_encoder):
    dataset = AutoRegressiveDataset.from_chunk(image_datachunk_with_encoder)
    for seq in dataset.data:
        assert list(seq.shape) == [100, 17]
    assert len(dataset) == 10

def test_autoregressive_dataset_pipeline2(cmd_datachunk):
    dataset = AutoRegressiveDataset.from_chunk(cmd_datachunk)
    for seq in dataset.data:
        assert list(seq.shape) == [20, 8]
    assert len(dataset) == 10

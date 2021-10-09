import numpy as np
import torch
from mimic.datatype import CommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.dataset import ReconstructionDataset
from mimic.dataset import AutoRegressiveDataset
import pytest

from test_datatypes import cmd_datachunk
from test_datatypes import image_datachunk
from test_datatypes import image_datachunk_with_encoder

def test_reconstruction_dataset_pipeline(image_datachunk):
    dataset = ReconstructionDataset.from_chunk(image_datachunk)
    assert list(dataset.data.shape) == [10 * 100, 3, 28, 28]

def test_autoregressive_dataset_pipeline1(image_datachunk_with_encoder):
    dataset = AutoRegressiveDataset.from_chunk(image_datachunk_with_encoder)
    for seq in dataset.data:
        assert list(seq.shape) == [100, 16]

def test_autoregressive_dataset_pipeline2(cmd_datachunk):
    dataset = AutoRegressiveDataset.from_chunk(cmd_datachunk)
    for seq in dataset.data:
        assert list(seq.shape) == [20, 7]


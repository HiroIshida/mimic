import numpy as np
import torch
from mimic.datatype import CommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.models import ImageAutoEncoder
from mimic.dataset import ReconstructionDataset
import pytest

@pytest.fixture(scope='session')
def reconstruction_dataset():
    n_seq = 100
    n_channel = 3
    n_pixel = 28
    ae = ImageAutoEncoder(16, torch.device('cpu'), image_shape=(n_channel, n_pixel, n_pixel))
    chunk = ImageDataChunk()
    for i in range(10):
        imgseq = np.random.randn(n_seq, n_channel, n_pixel, n_pixel)
        chunk.push_epoch(imgseq)
    dataset = ReconstructionDataset.from_chunk(chunk)
    return dataset

def test_reconstruction_dataset_pipeline(reconstruction_dataset):
    dataset = reconstruction_dataset
    assert list(dataset.data.shape) == [10 * 100, 3, 28, 28]

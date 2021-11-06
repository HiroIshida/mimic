import numpy as np
import os
import torch
import pybullet_data
from mimic.datatype import CommandDataChunk
from mimic.datatype import ImageDataChunk
from mimic.dataset import ReconstructionDataset
import mimic.dataset
from mimic.dataset import attach_flag_info
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset
from mimic.dataset import KinematicsDataset
import pytest

from test_datatypes import cmd_datachunk
from test_datatypes import image_datachunk
from test_datatypes import _img_chunk_uneven_n
from test_datatypes import image_datachunk_with_encoder
from test_datatypes import image_command_datachunk_with_encoder
from test_datatypes import auged_image_command_datachunk

def test_reconstruction_dataset_pipeline(image_datachunk):
    dataset = ReconstructionDataset.from_chunk(image_datachunk)
    assert list(dataset.data.shape) == [10 * 100 + _img_chunk_uneven_n, 3, 28, 28]
    assert len(dataset) == 10 * 100 + _img_chunk_uneven_n
    assert list(dataset[0].shape) == [3, 28, 28]

def test_attach_flag_info():
    seq1 = torch.randn((10, 3))
    seq2 = torch.randn((12, 3))
    seq3 = torch.randn((14, 3))
    seq_list = [seq1, seq2, seq3]
    seq_list_with_flag = attach_flag_info(seq_list)

    continue_flag = mimic.dataset._continue_flag
    end_flag = mimic.dataset._end_flag
    val_padding = mimic. dataset._val_padding
    assert seq_list_with_flag[0][9, 3] == continue_flag
    assert seq_list_with_flag[0][10, 3] == end_flag
    assert seq_list_with_flag[1][11, 3] == continue_flag
    assert seq_list_with_flag[1][12, 3] == end_flag
    assert seq_list_with_flag[2][13, 3] == continue_flag

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

def test_auged_autoregressive_dataset_pipeline(auged_image_command_datachunk):
    dataset = AutoRegressiveDataset.from_chunk(auged_image_command_datachunk)
    assert len(dataset) == 10
    sample_input, sample_output = dataset[0]
    assert list(sample_input.shape) == [100 - 1, 16 + 7 + 6 + 1]
    assert list(sample_output.shape) == [100 - 1, 16 + 7 + 6 + 1]

def test_biased_autoregressive_dataset_pipeline(image_command_datachunk_with_encoder):
    chunk = image_command_datachunk_with_encoder
    dataset = BiasedAutoRegressiveDataset.from_chunk(chunk)
    assert len(dataset) == 10
    assert dataset.n_bias == chunk.n_encoder_output()
    sample_input, sample_output = dataset[0]
    assert list(sample_input.shape) == [100 - 1, 16 + 7 + 1]
    assert list(sample_output.shape) == [100 - 1, 7 + 1]

def test_FirstOrderARDataset_pipeline(image_command_datachunk_with_encoder):
    chunk = image_command_datachunk_with_encoder
    dataset = FirstOrderARDataset.from_chunk(chunk)
    assert dataset.data_pre.ndim == 2
    assert list(dataset.data_pre.shape) == [10 * (100 - 1), 23]
    assert list(dataset.data_pre.shape) == list(dataset.data_pre.shape)

def test_BiasedFirstOrderARDataset_pipeline(image_command_datachunk_with_encoder):
    chunk = image_command_datachunk_with_encoder
    dataset = BiasedFirstOrderARDataset.from_chunk(chunk)
    assert dataset.data_pre.ndim == 2
    assert dataset.data_post.ndim == 2
    assert dataset.biases.ndim == 2
    assert len(dataset) == 10 * (100 -1)
    assert list(dataset.biases.shape) == [10 * (100 - 1), 16]
    assert list(dataset.data_pre.shape) == [10 * (100 - 1), 7]
    assert list(dataset.data_pre.shape) == list(dataset.data_pre.shape)

    pre, post, bias = dataset[0]
    assert list(pre.shape) == [7]
    assert list(post.shape) == [7]
    assert list(bias.shape) == [16]

@pytest.fixture(scope='session')
def kinematics_dataset():
    pbdata_path = pybullet_data.getDataPath()
    urdf_path = os.path.join(pbdata_path, 'kuka_iiwa', 'model.urdf')
    joint_names = ['lbr_iiwa_joint_{}'.format(idx+1) for idx in range(7)]
    link_names = ['lbr_iiwa_link_6', 'lbr_iiwa_link_7']
    dataset = KinematicsDataset.from_urdf(urdf_path, joint_names, link_names, n_sample=5)
    return dataset

def test_kinemanet_pipeline(kinematics_dataset):
    inp, out = kinematics_dataset[0]
    assert len(inp) == 7
    assert len(out) == 6 * 2


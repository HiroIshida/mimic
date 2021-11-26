import torch

from mimic.augmentation import augment_noisy_sequence
from mimic.augmentation import randomly_shrink_sequence
from mimic.augmentation import randomly_extend_sequence

from test_datatypes import cmd_datachunk

def test_data_augmentation(cmd_datachunk):
    n_data_aug = 10
    chunk = cmd_datachunk
    seq_list = chunk.to_featureseq_list()
    n_dim = seq_list[0].shape[1]
    cov = torch.eye(n_dim)

    seq_list_auged = augment_noisy_sequence(seq_list, cov, n_data_aug=n_data_aug)
    assert len(seq_list_auged) == len(seq_list) * n_data_aug
    for seq in seq_list_auged: 
        seq_shrinked = randomly_shrink_sequence(seq)
        assert len(seq_shrinked) <= len(seq)

        seq_extended = randomly_extend_sequence(seq, cov=cov)
        assert len(seq_extended) >= len(seq)


import numpy as np
import torch

from typing import Tuple, List
import logging
logger = logging.getLogger(__name__)

def compute_covariance_matrix(seqs_list: List[torch.Tensor]):
    diffs: List[torch.Tensor] = []
    for seqs in seqs_list:
        x_pre = seqs[:-1, :]
        x_post = seqs[1:, :]
        diff = x_post - x_pre
        diffs.append(diff)
    diffs_cat = torch.cat(diffs, dim=0)
    cov = torch.cov(diffs_cat.T)
    return cov

def augment_data(seqs_list: List[torch.Tensor], n_data_aug: int=10, cov_scale: float=0.3) -> List[torch.Tensor]:
    if n_data_aug < 1:
        logger.info("because n_data_aug < 1, skip data augmentation process..")
        return seqs_list
    logger.info("augment data with parmas: n_data_aug {0}, cov_scale {1}".format(
        n_data_aug, cov_scale))
    cov = compute_covariance_matrix(seqs_list) * cov_scale ** 2
    seqs_list = augment_noisy_sequence(seqs_list, cov, n_data_aug)

    # finally add deleted and extended sequences
    seqs_additional = []
    for seq in seqs_list:
        seq_shrinked = randomly_shrink_sequence(seq)
        seqs_additional.append(seq_shrinked)
        seq_extend = randomly_extend_sequence(seq_shrinked, cov)
        seqs_additional.append(seq_extend)
    seqs_list.extend(seqs_additional)
    return seqs_list

def augment_noisy_sequence(
        seqs_list: List[torch.Tensor], cov: torch.Tensor, n_data_aug: int=10) -> List[torch.Tensor]:
    cov_dim = cov.shape[0]
    walks_new = []
    for walk in seqs_list:
        n_seq, n_dim = walk.shape
        assert cov_dim == n_dim
        for _ in range(n_data_aug):
            rand_aug = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=cov, size=n_seq)
            assert rand_aug.shape == walk.shape
            walks_new.append(walk + torch.from_numpy(rand_aug).float())
    return walks_new

def randomly_shrink_sequence(seq: torch.Tensor) -> torch.Tensor:
    # choose delete index
    n_seq_len, _ = seq.shape
    idxes_delete = np.random.randint(n_seq_len, size=int(n_seq_len * np.random.rand() * 0.5))
    idxes_nondelete = set(np.arange(n_seq_len)) - set(idxes_delete)
    return seq[list(idxes_nondelete)]

def randomly_extend_sequence(seq: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    # insertion
    n_seq_len, n_dim = seq.shape
    idxes_insert = np.random.randint(n_seq_len, size=int(n_seq_len * np.random.rand() * 0.5))

    seq_new = []
    for i in range(n_seq_len):
        seq_new.append(seq[i]) 
        if i in set(idxes_insert):
            n_insert_len = np.random.randint(int(n_seq_len * 0.1))
            noises = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=cov, size=n_insert_len)
            for j in range(n_insert_len):
                seq_new.append(seq[i] + noises[j])
    return torch.stack(seq_new).float()


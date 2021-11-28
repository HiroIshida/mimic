from abc import ABC, abstractmethod
import random
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

def generate_insert_partitoin(n_seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    n_insert_total = np.random.randint(int(n_seq_len * 1.5))
    n_insert_list = []
    count = 0
    while True:
        n_insert = np.random.randint(int(n_seq_len * 0.2)) + 1
        if count + n_insert > n_insert_total:
            n_insert = n_insert_total - count
            if n_insert > 0:
                n_insert_list.append(n_insert)
            break
        n_insert_list.append(n_insert)
        count += n_insert
    idxes_insert = np.array(random.sample(range(n_seq_len), len(n_insert_list)))
    arg_idxes = np.argsort(idxes_insert)
    assert sum(n_insert_list) == n_insert_total
    return idxes_insert[arg_idxes], np.array(n_insert_list)[arg_idxes]

def augment_seq_data(seqs_list: List[torch.Tensor], n_data_aug: int, cov_scale: float, with_shrink_extend:bool = False) -> List[torch.Tensor]:
    if n_data_aug < 1:
        logger.info("because n_data_aug < 1, skip data augmentation process..")
        return seqs_list
    logger.info("augment data with parmas: n_data_aug {0}, cov_scale {1}".format(
        n_data_aug, cov_scale))
    cov = compute_covariance_matrix(seqs_list) * cov_scale ** 2
    seqs_list = augment_noisy_sequence(seqs_list, cov, n_data_aug)

    # finally add deleted and extended sequences
    if with_shrink_extend:
        raise NotImplementedError('Under construction. From experiences so far, this augmentation leads to worse results.')
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
    idxes_insert, n_insert_list = generate_insert_partitoin(n_seq_len)

    seq_new = []
    for i in range(n_seq_len):
        seq_new.append(seq[i]) 

        idx_insert_inner = np.where(idxes_insert==i)[0]
        not_found = len(idx_insert_inner) == 0
        if not_found: continue

        n_insert = n_insert_list[idx_insert_inner]
        noises = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=cov, size=n_insert)
        for noise in noises:
            seq_new.append(seq[i] + noise)
    return torch.stack(seq_new).float()

class AugmentationPost(ABC):
    @abstractmethod
    def __call__(self, seqs_list: List[torch.Tensor]) -> List[torch.Tensor]: ...

class DummyAugmentationPost(AugmentationPost):
    def __call__(self, seqs_list): return seqs_list

class SequenceAugmentation(AugmentationPost):
    n_data_aug: int
    cov_scale: float
    def __init__(self, n_data_aug: int=10, cov_scale: float=0.2):
        self.n_data_aug = n_data_aug
        self.cov_scale = cov_scale

    def __call__(self, seqs_list):
        return augment_seq_data(seqs_list, self.n_data_aug, self.cov_scale)

import torch
from mimic.models.common import LossDict, sum_loss_dict

def test_sum_loss_dict():
    loss_a1 = torch.tensor([1.0])
    loss_b1 = torch.tensor([2.0])
    loss_a2 = torch.tensor([3.0])
    loss_b2 = torch.tensor([4.0])

    dict1 = LossDict({'lossa': loss_a1, 'lossb': loss_b1})
    dict2 = LossDict({'lossa': loss_a2, 'lossb': loss_b2})

    dict_sum = sum_loss_dict([dict1, dict2])
    assert len(list(dict_sum.keys())) == 2
    assert dict_sum['lossa'] == 4.0
    assert dict_sum['lossb'] == 6.0

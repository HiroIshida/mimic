import sys
import torch
from mimic.models.common import LossDict
from mimic.models.common import LossDictNoGrad
from mimic.models.common import sum_loss_dict
from mimic.models.common import detach_clone

def test_detach_clone():
    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)
    P = 3*a**3 + b**2
    Q = 3*a**3 - b**2
    loss_dict = LossDict({'P': P, 'Q': Q})

    size_pre = sys.getsizeof(loss_dict)
    assert loss_dict['P'].requires_grad 
    assert loss_dict['Q'].requires_grad 

    loss_dict_new = detach_clone(loss_dict)
    size_post = sys.getsizeof(loss_dict)
    assert not loss_dict_new['P'].requires_grad
    assert not loss_dict_new['Q'].requires_grad

def test_sum_loss_dict():
    loss_a1 = torch.tensor([1.0])
    loss_b1 = torch.tensor([2.0])
    loss_a2 = torch.tensor([3.0])
    loss_b2 = torch.tensor([4.0])

    dict1 = LossDictNoGrad({'lossa': loss_a1, 'lossb': loss_b1})
    dict2 = LossDictNoGrad({'lossa': loss_a2, 'lossb': loss_b2})

    dict_sum = sum_loss_dict([dict1, dict2])
    assert len(list(dict_sum.keys())) == 2
    assert dict_sum['lossa'] == 4.0
    assert dict_sum['lossb'] == 6.0

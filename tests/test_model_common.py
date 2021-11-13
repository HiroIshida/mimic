import torch
from mimic.datatype import FeatureInfo
from mimic.models.common import NullConfig
from mimic.models.common import _PropModelConfigBase
from mimic.models import LSTMConfig
from mimic.models.common import LossDict
from mimic.models.common import LossDictFloat
from mimic.models.common import average_loss_dict
from mimic.models.common import to_scalar_values

def test_model_config():
    conf1 = NullConfig()
    assert len(conf1.hash_value) == 7

    conf2 = LSTMConfig(2)
    assert len(conf2.hash_value) == 7

    for key in FeatureInfo.__dataclass_fields__.keys():
        assert key in _PropModelConfigBase.__dict__.keys()

def test_to_scalar_values():
    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 4.], requires_grad=True)
    P = (3*a**3 + b**2).sum()
    Q = (3*a**3 - b**2).sum()
    loss_dict = LossDict({'P': P, 'Q': Q})

    assert loss_dict['P'].requires_grad 
    assert loss_dict['Q'].requires_grad 
    loss_dict_new = to_scalar_values(loss_dict)

def test_sum_loss_dict():
    loss_a1 = torch.tensor([1.0])
    loss_b1 = torch.tensor([2.0])
    loss_a2 = torch.tensor([3.0])
    loss_b2 = torch.tensor([4.0])

    dict1 = LossDictFloat({'lossa': loss_a1, 'lossb': loss_b1})
    dict2 = LossDictFloat({'lossa': loss_a2, 'lossb': loss_b2})

    dict_sum = average_loss_dict([dict1, dict2])
    assert len(list(dict_sum.keys())) == 2
    assert dict_sum['lossa'] == 2.0
    assert dict_sum['lossb'] == 3.0


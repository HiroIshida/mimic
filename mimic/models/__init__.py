from functools import lru_cache
from typing import Dict

from .common import NullConfig
from .autoencoder import AbstractEncoderDecoder
from .autoencoder import ImageAutoEncoder
from .lstm import LSTMConfig
from .lstm import BiasedLSTMConfig
from .lstm import AugedLSTMConfig
from .lstm import LSTMBase
from .lstm import LSTM
from .lstm import BiasedLSTM
from .lstm import AugedLSTM
from .denseprop import DenseConfig
from .denseprop import DeprecatedDenseConfig
from .denseprop import BiasedDenseConfig
from .denseprop import KinemaNetConfig
from .denseprop import DenseBase
from .denseprop import DenseProp
from .denseprop import DeprecatedDenseProp
from .denseprop import BiasedDenseProp
from .denseprop import KinemaNet

from .common import _Model

@lru_cache(maxsize=None)
def create_model_dispath_table() -> Dict:
    table = {}
    stack = [_Model]
    while len(stack)>0:
        cls_here = stack.pop()
        for cls_child in cls_here.__subclasses__():
            name = cls_child.__name__
            table[name] = cls_child
            stack.append(cls_child)
    return table

def get_model_type_from_name(name: str) -> _Model:
    table = create_model_dispath_table()
    return table[name]

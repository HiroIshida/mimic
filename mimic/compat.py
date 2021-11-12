from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Subset
from mimic.models.common import _Model
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import DenseProp
from mimic.models import BiasedDenseProp
from mimic.models import DeprecatedDenseProp
from mimic.models import KinemaNet
from mimic.models import AugedLSTM

from mimic.dataset import _DatasetFromChunk
from mimic.dataset import ReconstructionDataset
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset
from mimic.dataset import AugedAutoRegressiveDataset
from mimic.dataset import KinematicsDataset

import typing
from typing import Type
from typing import Dict
from typing import Union

# The reason why compatibility is not written as a class variable of a model is that 
# we do not intriduce a hierarchy between models and dataset. 
# IF model.compatible_dataset = HogeData, then model can be defined only after/upon the 
# definition of dataset

# by declaring this table as global variable, leave a room for user to modify or add some
# custom compatibility relations 

# TODO(HiroIshida) maybe values are list
_DatasetT = Union[Type[_DatasetFromChunk], Type[KinematicsDataset]]
_dataset_compat_table : Dict[str, _DatasetT] = {
        ImageAutoEncoder.__name__: ReconstructionDataset,
        LSTM.__name__: AutoRegressiveDataset,
        AugedLSTM.__name__: AugedAutoRegressiveDataset,
        BiasedLSTM.__name__: BiasedAutoRegressiveDataset,
        DenseProp.__name__: AutoRegressiveDataset,
        BiasedDenseProp.__name__: BiasedAutoRegressiveDataset,
        DeprecatedDenseProp.__name__: FirstOrderARDataset,
        KinemaNet.__name__: KinematicsDataset
        }

def get_compat_dataset_type(model: _Model) -> _DatasetT:
    return _dataset_compat_table[model.__class__.__name__]

@typing.no_type_check 
def is_compatible(model: _Model, dataset: Dataset) -> bool:
    compat_dataset = get_compat_dataset_type(model)
    compat_dataset_name = compat_dataset.__name__
    if isinstance(dataset, Subset):
        return dataset.dataset.__class__.__name__ == compat_dataset_name
    return dataset.__class__.__name__ == compat_dataset_name

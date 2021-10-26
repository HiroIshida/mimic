from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Subset
from mimic.models.common import _Model
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import DenseProp
from mimic.models import BiasedDenseProp

from mimic.dataset import _Dataset
from mimic.dataset import ReconstructionDataset
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset

import typing
from typing import Type

# The reason why compatibility is not written as a class variable of a model is that 
# we do not intriduce a hierarchy between models and dataset. 
# IF model.compatible_dataset = HogeData, then model can be defined only after/upon the 
# definition of dataset

# by declaring this table as global variable, leave a room for user to modify or add some
# custom compatibility relations 

# TODO(HiroIshida) maybe values are list
_dataset_compat_table = {
        ImageAutoEncoder.__name__: [ReconstructionDataset],
        LSTM.__name__: [AutoRegressiveDataset],
        BiasedLSTM.__name__: [BiasedAutoRegressiveDataset],
        DenseProp.__name__: [FirstOrderARDataset],
        BiasedDenseProp.__name__: [BiasedFirstOrderARDataset]
        }

@typing.no_type_check 
def is_compatible(model: _Model, dataset: Dataset) -> bool:
    compat_models = _dataset_compat_table[model.__class__.__name__]
    compat_model_names = [m.__name__ for m in compat_models]
    if isinstance(dataset, Subset):
        return dataset.dataset.__class__.__name__ in compat_model_names
    return dataset.__class__.__name__ in compat_model_names

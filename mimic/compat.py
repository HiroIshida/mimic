from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import Subset
from mimic.models.common import _Model
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.models import DenseProp
from mimic.models import BiasedDenseProp

from mimic.dataset import ReconstructionDataset
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import FirstOrderARDataset
from mimic.dataset import BiasedFirstOrderARDataset

from typing import Type

# The reason why compatibility is not written as a class variable of a model is that 
# we do not intriduce a hierarchy between models and dataset. 
# IF model.compatible_dataset = HogeData, then model can be defined only after/upon the 
# definition of dataset

# by declaring this table as global variable, leave a room for user to modify or add some
# custom compatibility relations 

# TODO(HiroIshida) maybe values are list
_dataset_compat_table = {
        ImageAutoEncoder.__name__: ReconstructionDataset.__name__,
        LSTM.__name__: AutoRegressiveDataset.__name__,
        DenseProp.__name__: FirstOrderARDataset.__name__,
        BiasedDenseProp.__name__: BiasedFirstOrderARDataset.__name__
        }

def compatible_dataset(model: _Model) -> str:
    model_type_name = model.__class__.__name__
    return _dataset_compat_table[model_type_name]

def is_compatible(model: _Model, dataset: Dataset) -> bool:
    if isinstance(dataset, Subset):
        return compatible_dataset(model) == dataset.dataset.__class__.__name__ 
    return compatible_dataset(model) == dataset.__class__.__name__

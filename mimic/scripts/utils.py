# type: ignore
# note: Dataset doesn't have default __len__, but user must implement in any case (type ignore)
from torch.utils.data import Dataset
from torch.utils.data import random_split

def split_with_ratio(dataset: Dataset, valid_raio: float=0.1):
    n_total = len(dataset) 
    n_validate = int(0.1 * n_total)
    ds_train, ds_validate = random_split(dataset, [n_total-n_validate, n_validate])  
    return ds_train, ds_validate

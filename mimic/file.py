import os
import os.path as osp
import pickle
from typing import Any
from typing import Type
from typing import TypeVar
from typing import Optional

def get_data_dir() -> str:
    dirname = osp.expanduser('~/.mimic')
    if not osp.exists(dirname):
        os.makedirs(dirname)
    return dirname

def get_project_dir(project_name: str) ->str:
    dirname = osp.join(get_data_dir(), project_name)
    if not osp.exists(dirname):
        os.makedirs(dirname)
    return dirname

def _cache_name(project_name: str, cls: type, prefix: Optional[str] = None) -> str:
    filename = cls.__name__
    if prefix:
        filename = prefix + filename
    wholename = osp.join(get_project_dir(project_name), filename)
    return wholename

DataT = TypeVar('DataT') 
def load_pickled_data(project_name: str, cls: Type[DataT], prefix: Optional[str] = None) -> DataT:
    wholename = _cache_name(project_name, cls, prefix)
    with open(wholename, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_pickled_data(data: Any, project_name: str, prefix: Optional[str] = None) -> None:
    wholename = _cache_name(project_name, data.__class__, prefix)
    with open(wholename, 'wb') as f:
        pickle.dump(data, f)

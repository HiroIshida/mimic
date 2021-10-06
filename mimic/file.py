import os
import os.path as osp
import pickle
from typing import Any

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

def load_pickled_data(project_name: str, cls: type) -> Any:
    filename = osp.join(get_project_dir(project_name), cls.__name__)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_pickled_data(data: Any, project_name: str, cls: type) -> None:
    filename = osp.join(get_project_dir(project_name), cls.__name__)
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

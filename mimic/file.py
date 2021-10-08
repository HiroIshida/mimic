import os
import os.path as osp
import pickle
from typing import Any
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

def load_pickled_data(project_name: str, cls: type) -> Any:
    filename = osp.join(get_project_dir(project_name), cls.__name__)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_pickled_data(data: Any, project_name: str, post_fix: Optional[str] = None) -> None:
    filename = data.__class__.__name__
    if post_fix:
        filename += ('_' + post_fix)
    wholename = osp.join(get_project_dir(project_name), filename)
    with open(wholename, 'wb') as f:
        pickle.dump(data, f)

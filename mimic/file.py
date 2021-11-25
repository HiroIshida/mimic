import os
import os.path as osp
import re
import pickle
from typing import Any
from typing import List
from typing import Type
from typing import TypeVar
from typing import Optional

import logging
logger = logging.getLogger(__name__)

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

def _cache_name(project_name: str, cls: type, 
        prefix: Optional[str] = None, postfix: Optional[str] = None) -> str:
    filename = cls.__name__
    if prefix:
        filename = prefix + filename
    if postfix: 
        filename = filename + postfix
    wholename = osp.join(get_project_dir(project_name), filename)
    return wholename

def _cache_name_list(project_name: str, cls: type, 
        prefix: Optional[str] = None, postfix: Optional[str] = None) -> List[str]:
    base_name = _cache_name(project_name, cls, prefix, postfix)

    cache_name_list = []
    head, tail = os.path.split(base_name)
    fnames = os.listdir(head)
    for fname in fnames:
        res = re.match(r'{}*.'.format(tail), fname)
        if res is not None:
            whole_name = os.path.join(head, fname)
            cache_name_list.append(whole_name)
    return cache_name_list

DataT = TypeVar('DataT') 
def load_pickled_data(project_name: str, cls: Type[DataT], 
        prefix: Optional[str] = None, postfix: Optional[str] = None) -> List[DataT]:
    filenames = _cache_name_list(project_name, cls, prefix, postfix)

    data_list = []
    for fname in filenames:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        data_list.append(data)
    return data_list

def dump_pickled_data(data: Any, project_name: str, 
        prefix: Optional[str] = None, postfix: Optional[str] = None) -> None:
    wholename = _cache_name(project_name, data.__class__, prefix, postfix)
    logger.info('dump pickle to {}'.format(wholename))

    with open(wholename, 'wb') as f:
        pickle.dump(data, f)

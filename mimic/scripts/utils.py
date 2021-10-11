# note: Dataset doesn't have default __len__, but user must implement in any case (type ignore)
import time
import os
import logging
from logging import Logger

from torch.utils.data import Dataset
from torch.utils.data import random_split

from mimic.file import get_project_dir

def split_with_ratio(dataset: Dataset, valid_raio: float=0.1):
    n_total = len(dataset) # type: ignore
    n_validate = int(0.1 * n_total)
    ds_train, ds_validate = random_split(dataset, [n_total-n_validate, n_validate])  
    return ds_train, ds_validate

def create_default_logger(project_name: str, prefix: str) -> Logger:
    timestr = "_" + time.strftime("%Y%m%d%H%M%S")
    log_file_name = os.path.join(get_project_dir(project_name), (prefix + timestr + '.log'))
    FORMAT = '[%(levelname)s] %(asctime)s %(name)s: %(message)s'
    logging.basicConfig(filename=log_file_name, format=FORMAT)
    logger = logging.getLogger('mimic')
    logger.setLevel(level=logging.INFO)

    log_sym_name = os.path.join(get_project_dir(project_name), ('latest_' + prefix + '.log'))
    logger.info('create log symlink :{0} => {1}'.format(log_file_name, log_sym_name))
    if os.path.islink(log_sym_name):
        os.unlink(log_sym_name)
    os.symlink(log_file_name, log_sym_name)
    return logger

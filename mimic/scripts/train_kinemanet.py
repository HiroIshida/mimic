import argparse
import os
import pybullet_data
import torch

from mimic.dataset import KinematicsDataset
from mimic.models import KinemaNet, DenseConfig, KinemaNetConfig
from mimic.trainer import train
from mimic.trainer import Config
from mimic.trainer import TrainCache
from mimic.scripts.utils import split_with_ratio
from mimic.scripts.utils import create_default_logger

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kinematics', help='project name')
    parser.add_argument('-rn', type=str, default='kuka', help='robot_name')
    parser.add_argument('-n', type=int, default=-1, help='epoch')
    parser.add_argument('-m', type=int, default=-1, help='sample')

    args = parser.parse_args()
    project_name = args.pn
    robot_name = args.rn
    n_epoch = args.n
    n_sample = None if args.m == -1 else args.m

    if robot_name == 'kuka':
        pbdata_path = pybullet_data.getDataPath()
        urdf_path = os.path.join(pbdata_path, 'kuka_iiwa', 'model.urdf')
        joint_names = ['lbr_iiwa_joint_{}'.format(idx+1) for idx in range(7)]
        link_names = ['lbr_iiwa_link_7']
    else:
        raise Exception

    logger = create_default_logger(project_name, 'kinemanet_{}'.format(robot_name))

    dataset = KinematicsDataset.from_urdf(urdf_path, joint_names, link_names, n_sample=n_sample)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KinemaNet(device, dataset.meta_data, KinemaNetConfig(200, 6))

    ds_train, ds_valid = split_with_ratio(dataset)
    tcache = TrainCache[KinemaNet](project_name, KinemaNet, cache_postfix='_' + robot_name)
    config = Config(batch_size=1000, n_epoch=n_epoch) 
    train(model, ds_train, ds_valid, tcache=tcache, config=config)

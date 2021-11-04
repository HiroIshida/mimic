import argparse

import torch
from mimic.datatype import ImageDataChunk
from mimic.datatype import ImageCommandDataChunk
from mimic.dataset import ReconstructionDataset
from mimic.models import ImageAutoEncoder
from mimic.trainer import train
from mimic.trainer import Config
from mimic.trainer import TrainCache
from mimic.scripts.utils import split_with_ratio
from mimic.scripts.utils import create_default_logger
from mimic.scripts.utils import query_yes_no

def train_auto_encoder(project_name: str, n_bottleneck: int, config: Config) -> None:
    try:
        tmp = ImageCommandDataChunk.load(project_name)
        chunk_ = ImageDataChunk.from_imgcmd_chunk(tmp)
    except FileNotFoundError:
        chunk_ = ImageDataChunk.load(project_name)
    n_intact = 5
    _, chunk = chunk_.split(n_intact)
    dataset = ReconstructionDataset.from_chunk(chunk)
    ds_train, ds_valid = split_with_ratio(dataset)
    image_shape = dataset[0].shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageAutoEncoder(device, n_bottleneck, image_shape=image_shape)
    tcache = TrainCache[ImageAutoEncoder](project_name, ImageAutoEncoder)
    train(model, ds_train, ds_valid, tcache=tcache, config=config)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=1000, help='training epoch')
    parser.add_argument('-bottleneck', type=int, default=16, help='latent dimension')
    args = parser.parse_args()
    project_name = args.pn
    n_epoch = args.n
    n_bottleneck = args.bottleneck

    logger = create_default_logger(project_name, 'autoencoder')
    config = Config(n_epoch=n_epoch)
    train_auto_encoder(project_name, n_bottleneck, config)

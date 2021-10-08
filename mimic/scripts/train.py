import argparse
import torch

from mimic.datatype import ImageDataChunk
from mimic.dataset import ReconstructionDataset
from mimic.models import ImageAutoEncoder
from mimic.trainer import train
from mimic.trainer import Config
from mimic.scripts.utils import split_with_ratio

def train_auto_encoder(project_name: str, n_bottleneck: int, config: Config) -> None:
    chunk = ImageDataChunk.load(project_name)
    dataset = ReconstructionDataset.from_chunk(chunk)
    ds_train, ds_valid = split_with_ratio(dataset)
    image_shape = dataset[0].shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageAutoEncoder(n_bottleneck, device, image_shape=image_shape)
    train(model, ds_train, ds_valid, config=config)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=1000, help='training epoch')
    parser.add_argument('-bottleneck', type=int, default=16, help='latent dimension')
    args = parser.parse_args()
    project_name = args.pn
    n_epoch = args.n
    n_bottleneck = args.bottleneck
    config = Config(n_epoch=n_epoch)
    train_auto_encoder(project_name, n_bottleneck, config)


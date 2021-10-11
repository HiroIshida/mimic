import argparse

from typing import List
import numpy as np
import torch

from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.predictor import ImageLSTMPredictor
from mimic.trainer import TrainCache

def create_predictor(project_name: str) -> ImageLSTMPredictor:
    ae_train_cache = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
    lstm_train_cache = TrainCache[LSTM].load(project_name, LSTM)
    return ImageLSTMPredictor(ae_train_cache.best_model, lstm_train_cache.best_model)

if __name__=='__main__':
    # only for demo
    #import moviepy
    from mimic.datatype import ImageDataChunk
    from mimic.datatype import ImageDataSequence
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=1000, help='training epoch')
    parser.add_argument('-bottleneck', type=int, default=16, help='latent dimension')

    print('supposed to be used only in testing or debugging...')

    args = parser.parse_args()
    project_name = args.pn
    predictor = create_predictor(project_name)
    chunk = ImageDataChunk.load(project_name)
    seq: ImageDataSequence = chunk[-1][ImageDataSequence]
    assert seq.data.ndim == 4
    for i in range(20):
        predictor.feed(seq.data[i])
    imgs = predictor.predict(60)

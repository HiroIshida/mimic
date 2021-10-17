import argparse

from typing import List
from typing import Union
import numpy as np
import torch

from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.predictor import ImageCommandLSTMPredictor
from mimic.trainer import TrainCache

def create_predictor(project_name: str) -> ImageCommandLSTMPredictor:
    ae_train_cache = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
    lstm_train_cache = TrainCache[LSTM].load(project_name, LSTM)
    return ImageCommandLSTMPredictor(lstm_train_cache.best_model, ae_train_cache.best_model)

if __name__=='__main__':
    # only for demo
    import os
    from moviepy.editor import ImageSequenceClip
    from mimic.file import get_project_dir
    from mimic.datatype import ImageDataSequence
    from mimic.datatype import CommandDataSequence
    from mimic.datatype import ImageCommandDataChunk
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-n', type=int, default=1000, help='training epoch')
    parser.add_argument('-bottleneck', type=int, default=16, help='latent dimension')

    print('supposed to be used only in testing or debugging...')

    args = parser.parse_args()
    project_name = args.pn
    predictor = create_predictor(project_name)
    chunk = ImageCommandDataChunk.load(project_name)
    imgseq: ImageDataSequence = chunk[-2][ImageDataSequence]
    cmdseq: CommandDataSequence = chunk[-2][CommandDataSequence]
    assert imgseq.data.ndim == 4
    for i in range(30):
        predictor.feed((imgseq.data[i], cmdseq.data[i]))
    imgseq_pred, cmdseq_pred = map(list, zip(*predictor.predict(30)))
    filename = os.path.join(get_project_dir(project_name), "prediction_result.gif")
    clip = ImageSequenceClip(imgseq_pred, fps=50)
    clip.write_gif(filename, fps=50)

import argparse

import typing
from typing import List
from typing import Union
import numpy as np
import torch

from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import AugedLSTM
from mimic.models import DenseProp
from mimic.models import BiasedDenseProp
from mimic.models import DeprecatedDenseProp
from mimic.predictor import AbstractPredictor
from mimic.predictor import ImageCommandPredictor
from mimic.predictor import FFImageCommandPredictor
from mimic.predictor import create_predictor
from mimic.trainer import TrainCache

@typing.no_type_check 
def create_predictor_from_name(project_name: str, model_name: str) -> \
        Union[ImageCommandPredictor, FFImageCommandPredictor]:
    dispatch_dict = {
            'lstm': LSTM,
            'biased_lstm': BiasedLSTM,
            'auged_lstm': AugedLSTM,
            'dense_prop': DenseProp,
            'biased_dense_prop': BiasedDenseProp,
            'depre_dense_prop': DeprecatedDenseProp,
            }
    ModelT = dispatch_dict[model_name]
    ae_train_cache = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)
    prop_train_cache = TrainCache[ModelT].load(project_name, ModelT)
    return create_predictor(ae_train_cache.best_model, prop_train_cache.best_model)

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
    parser.add_argument('-n', type=int, default=50, help='prediction length')
    parser.add_argument('-model', type=str, default='lstm', help='propagator model name')
    parser.add_argument('-bottleneck', type=int, default=16, help='latent dimension')

    args = parser.parse_args()
    project_name = args.pn
    model_name = args.model
    n_prediction = args.n

    predictor = create_predictor_from_name(project_name, model_name)
    chunk = ImageCommandDataChunk.load(project_name)
    n_intact = 5
    chunk_intact, _ = chunk.split(n_intact)
    imgseq, cmdseq = chunk_intact[0]
    assert imgseq.data.ndim == 4
    for i in range(n_prediction):
        predictor.feed((imgseq.data[i], cmdseq.data[i]))
    imgseq_pred, cmdseq_pred = map(list, zip(*predictor.predict(n_prediction)))
    filename = os.path.join(get_project_dir(project_name), 'prediction_result_{}.gif'.format(model_name))
    if isinstance(predictor, FFImageCommandPredictor):
        print('images can not be predicted by biased model') 
    else:
        clip = ImageSequenceClip(imgseq_pred, fps=50)
        clip.write_gif(filename, fps=50)

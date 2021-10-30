# mypy: ignore-errors
import argparse
from mimic.models import ImageAutoEncoder
from mimic.models import LSTM
from mimic.models import BiasedLSTM
from mimic.models import BiasedDenseProp
from mimic.dataset import AutoRegressiveDataset
from mimic.dataset import BiasedAutoRegressiveDataset
from mimic.compat import get_compat_dataset_type
from mimic.trainer import TrainCache
from mimic.predictor import evaluate_command_prediction_error
from mimic.scripts.train_propagator import prepare_trained_image_chunk

_model_table = {
        'lstm': LSTM,
        'biased_lstm': BiasedLSTM,
        'biased_dense_prop': BiasedDenseProp
        }

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    parser.add_argument('-model', type=str, default='lstm', help='propagator model name')

    args = parser.parse_args()
    project_name = args.pn
    model_name = args.model

    chunk = prepare_trained_image_chunk(project_name)
    n_intact = 5
    chunk_intact, _ = chunk.split(n_intact)
    modelT = _model_table[model_name]
    prop_train_cache = TrainCache.load(project_name, modelT)

    DatasetT = get_compat_dataset_type(prop_train_cache.best_model)
    prop_dataset = DatasetT.from_chunk(chunk_intact)

    ae_train_cache = TrainCache[ImageAutoEncoder].load(project_name, ImageAutoEncoder)

    val = evaluate_command_prediction_error(
            ae_train_cache.best_model,
            prop_train_cache.best_model,
            prop_dataset)
    print("command prediction error of {0}: {1}".format(model_name, val))

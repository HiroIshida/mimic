#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
from dataclasses import dataclass

@dataclass
class Config:
    project_name: str
    n_data: int
    n_train_ae: int
    n_train_lstm: int
    n_train_dense: int
    n_train_biased_dense: int

parser = argparse.ArgumentParser()
parser.add_argument('--dryrun', action='store_true', help='check pipeline only')
args = parser.parse_args()

dryrun = args.dryrun
if dryrun:
    cfg = Config('kuka_reaching_test', 10, 3, 3, 3, 3)
else:
    cfg = Config('kuka_reaching', 300, 1000, 4000, 2000, 2000)

here = os.path.dirname(os.path.realpath(sys.argv[0]))
script_path = os.path.join(here, 'kuka_reaching.py')

cmd_generate_dataset = 'python3 {0} -pn {1} -n {2}'.format(script_path, cfg.project_name, cfg.n_data)
cmd_train_autoencoder = 'python3 -m mimic.scripts.train_auto_encoder -pn {0} -n {1}'.format(cfg.project_name, cfg.n_train_ae)
cmd_train_lstm = 'python3 -m mimic.scripts.train_propagator -pn {0} -n {1}'.format(cfg.project_name, cfg.n_train_lstm)
cmd_train_dense_prop = 'python3 -m mimic.scripts.train_propagator -pn {0} -n {1} -model dense_prop'.format(cfg.project_name, cfg.n_train_dense)
cmd_train_biasd_dense_prop = 'python3 -m mimic.scripts.train_propagator -pn {0} -n {1} -model biased_dense_prop'.format(cfg.project_name, cfg.n_train_biased_dense)
cmd_run_predictor_lstm = 'python3 -m mimic.scripts.predict -pn {0} -model lstm'.format(cfg.project_name)
cmd_run_predictor_dense_prop = 'python3 -m mimic.scripts.predict -pn {0} -model dense_prop'.format(cfg.project_name)
cmd_run_predictor_biased_dense_prop = 'python3 -m mimic.scripts.predict -pn {0} -model biased_dense_prop'.format(cfg.project_name)

subprocess.check_call(cmd_generate_dataset, shell=True)
subprocess.check_call(cmd_train_autoencoder, shell=True)
subprocess.check_call(cmd_train_lstm, shell=True)
subprocess.check_call(cmd_train_dense_prop, shell=True)
subprocess.check_call(cmd_train_biasd_dense_prop, shell=True)
subprocess.check_call(cmd_run_predictor_lstm, shell=True)
subprocess.check_call(cmd_run_predictor_dense_prop, shell=True)
subprocess.check_call(cmd_run_predictor_biased_dense_prop, shell=True)

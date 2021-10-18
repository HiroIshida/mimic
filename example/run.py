#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
from dataclasses import dataclass

@dataclass
class Config:
    n_data: int
    n_train_ae: int
    n_train_lstm: int
    n_train_dense: int

parser = argparse.ArgumentParser()
parser.add_argument('--dryrun', action='store_true', help='check pipeline only')
args = parser.parse_args()

dryrun = args.dryrun
if dryrun:
    cfg = Config(10, 3, 10, 10)
else:
    cfg = Config(300, 3000, 4000, 2000)

here = os.path.dirname(os.path.realpath(sys.argv[0]))
script_path = os.path.join(here, 'kuka_reaching.py')
project_name = 'kuka_reaching'

cmd_generate_dataset = 'python3 {0} -n {1}'.format(script_path, cfg.n_data)
cmd_train_autoencoder = 'python3 -m mimic.scripts.train_auto_encoder -pn {0} -n {1}'.format(project_name, cfg.n_train_ae)
cmd_train_lstm = 'python3 -m mimic.scripts.train_propagator -pn {0} -n {1}'.format(project_name, cfg.n_train_lstm)
cmd_train_denseprop = 'python3 -m mimic.scripts.train_propagator -pn {0} -n {1} --dense'.format(project_name, cfg.n_train_dense)
cmd_run_predictor = 'python3 -m mimic.scripts.predict -pn {0}'.format(project_name)

subprocess.check_call(cmd_generate_dataset, shell=True)
subprocess.check_call(cmd_train_autoencoder, shell=True)
subprocess.check_call(cmd_train_lstm, shell=True)
subprocess.check_call(cmd_train_denseprop, shell=True)

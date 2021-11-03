import argparse
import os
import re
import pickle
import matplotlib.pyplot as plt
from mimic.file import get_project_dir
from mimic.trainer import TrainCache

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', type=str, default='kuka_reaching', help='project name')
    args = parser.parse_args()
    project_name = args.pn

    project_dir = get_project_dir(project_name)
    matplotlib_dir = os.path.join(project_dir, 'matplotlib')
    if not os.path.exists(matplotlib_dir):
        os.makedirs(matplotlib_dir)

    fnames = os.listdir(project_dir)
    for fname in fnames:
        m = re.match(r'.*TrainCache.*', fname)
        if m is not None:
            pickle_file = os.path.join(project_dir, fname)

            # TODO NOTE Directly loading pickle file is strange..
            # should use mimic.file functions in the future...
            with open(pickle_file, 'rb') as f:
                tcache: TrainCache = pickle.load(f)
                fig, ax = plt.subplots()
                tcache.visualize((fig, ax))
                image_file = os.path.join(matplotlib_dir, fname + '.png')
                fig.savefig(image_file)
                print('saved to {}'.format(image_file))

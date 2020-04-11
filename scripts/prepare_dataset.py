import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from utils import check_dir


def from_dataset_to_tensors(runs_dir, train_indices, test_indices):
    """
    :param runs_dir: directory containing the simulations
    :param train_indices
    :param test_indices
    :return: tensors with input and output of the network for both train and test set.
    """
    train_sample = []
    test_sample = []
    train_target = []
    test_target = []

    for file_name in os.listdir(runs_dir):
        pickle_file = os.path.join(runs_dir, file_name)
        run = pd.read_pickle(pickle_file)

        i = int(re.findall('\d+', file_name)[0])

        if i in train_indices:
            input = train_sample
            output = train_target
        else:
            input = test_sample
            output = test_target

        for step in run:
            for myt in step:
                # The input is the prox_values, that are the response values of ​​the sensors [array of 7 floats]
                sample = myt['prox_values'].copy()
                input.append(sample)

                # The output is the speed of the wheels (which we assume equals left and right) [array of 1 float]
                speed = myt['motor_left_target']
                output.append([speed])

    # Generate the tensors
    train_sample_tensor = torch.tensor(train_sample)
    test_sample_tensor = torch.tensor(test_sample)
    train_target_tensor = torch.tensor(train_target)
    test_target_tensor = torch.tensor(test_target)

    return train_sample_tensor, test_sample_tensor, train_target_tensor, test_target_tensor


def main(file, runs_dir):
    """
    :param file: file containing the defined indices for the split
    :param runs_dir: directory containing the simulations
    """
    # Uncomment the following line to generate a new dataset split
    # dataset_split(file)

    # Load the indices
    dataset = np.load(file)
    n_train = 800
    train_indices, test_indices = dataset[:n_train], dataset[n_train:]

    # Split the dataset also defining input and output, using the indices
    x_train, x_test, y_train, y_test = from_dataset_to_tensors(runs_dir, train_indices, test_indices)


if __name__ == '__main__':

    out_dir = 'models/distributed/'
    check_dir(out_dir)
    file = os.path.join(out_dir, 'dataset_split.npy')

    runs_dir = 'out-5myts/'

    try:
        main(file, runs_dir)
    except:
        raise

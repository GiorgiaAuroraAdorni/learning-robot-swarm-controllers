import os
import re
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from networks.distributed_network import DistributedNet
from torch.utils import data
from torch.utils.data import TensorDataset
from tqdm import tqdm
from utils import check_dir, plot_setting


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
        elif i in test_indices:
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


def unmask(label):
    """

    :param label:
    :return: torch.stack(new_label)
    """
    new_label = []

    for i in range(label.shape[0]):
        indices = np.where(label[i] < 0)
        new_label.append(np.delete(label[i], indices, axis=0))

    return torch.stack(new_label)


def train_net(epochs: int,
              train_dataset: data.TensorDataset,
              test_dataset: data.TensorDataset,
              net: torch.nn.Module,
              batch_size: int = 100,
              learning_rate: float = 0.01,
              training_loss: Optional[List[float]] = None,
              testing_loss: Optional[List[float]] = None,
              criterion=torch.nn.MSELoss(),
              padded=False,
              ) -> Tuple[List[float], List[float]]:
    """

    :param epochs:
    :param train_dataset:
    :param test_dataset:
    :param net:
    :param batch_size:
    :param learning_rate:
    :param training_loss:
    :param testing_loss:
    :param criterion:
    :param padded:
    :return training_loss, testing_loss:
    """

    train_minibatch = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_minibatch = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    optimizer.zero_grad()

    if training_loss is None:
        training_loss = []

    if testing_loss is None:
        testing_loss = []

    for _ in tqdm(range(epochs)):
        epoch_loss = 0.0

        for n, (inputs, labels) in enumerate(train_minibatch):
            output = net(inputs)

            # padded is used when in the simulations are different number of thymio
            if padded:
                losses = []
                for out, label in zip(output, labels):
                    label = unmask(label)
                    loss = criterion(out, label)
                    losses.append(loss)
                loss = torch.mean(torch.stack(losses))
            else:
                loss = criterion(output, labels)

            loss.backward()
            epoch_loss += float(loss)
            optimizer.step()
            optimizer.zero_grad()

        training_loss.append(epoch_loss)

        with torch.no_grad():
            # padded is used when in the simulations are different number of thymio
            if padded:
                test_losses = []
                for t_inputs, t_labels in test_minibatch:
                    t_output = net(t_inputs)
                    losses = []
                    for out, label in zip(t_output, t_labels):
                        label = unmask(label)
                        loss = criterion(out, label)
                        losses.append(loss)
                    loss = torch.mean(torch.stack(losses))
                    test_losses.append(float(loss))
                testing_loss.append(sum(test_losses))
            else:
                testing_loss.append(sum([float(criterion(net(inputs), labels)) for inputs, labels in test_minibatch]))

        print(epoch_loss)

    return training_loss, testing_loss


def main(file, runs_dir, out_dir, model):
    """
    :param file: file containing the defined indices for the split
    :param runs_dir: directory containing the simulations
    :param out_dir:
    :param model:
    """
    # Uncomment the following line to generate a new dataset split
    # dataset_split(file)

    # Load the indices
    dataset = np.load(file)
    n_train = 800
    train_indices, test_indices = dataset[:n_train], dataset[n_train:]

    # Split the dataset also defining input and output, using the indices
    x_train, x_test, y_train, y_test = from_dataset_to_tensors(runs_dir, train_indices, test_indices)

    d_training_set = TensorDataset(x_train, y_train)
    d_test_set = TensorDataset(x_test, y_test)

    command_dis = True
    save_cmd = True
    # save_cmd = False
    model = 'net1'

    if len(sys.argv) > 1:
        if sys.argv[2] == 'load':
            command_dis = False
        if sys.argv[4] == 'save':
            save_cmd = True

    if command_dis:
        d_net = DistributedNet(x_train.shape[1])
        d_training_loss, d_testing_loss = [], []

        training_loss, testing_loss = train_net(epochs=2, net=d_net, train_dataset=d_training_set,
                                                test_dataset=d_test_set, batch_size=100, learning_rate=0.001,
                                                training_loss=d_training_loss, testing_loss=d_testing_loss)

        print('training_loss %s, testing_loss %s.' % (training_loss, testing_loss))
        plot_setting(training_loss, testing_loss, out_dir, model)

        if save_cmd:
            torch.save(d_net, '%s/%s' % (out_dir, model))
    else:
        u = 1
        d_net = torch.load('%s/%s' % (out_dir, model))


if __name__ == '__main__':

    out_dir = 'models/distributed/'
    check_dir(out_dir)
    file = os.path.join(out_dir, 'dataset_split.npy')

    runs_dir = 'out-5myts/'
    model = 'net1'

    try:
        main(file, runs_dir, out_dir, model)
    except:
        raise

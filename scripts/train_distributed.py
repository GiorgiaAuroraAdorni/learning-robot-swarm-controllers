import os
import re
import sys
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
from tqdm import tqdm

from networks.distributed_network import DistributedNet
from utils import check_dir, plot_train_test_losses, scatter_plot, plot_prediction_histogram


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
        if not file_name.endswith('.pkl'):
            continue

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

    #  Generate a scatter plot to check the conformity of the dataset
    title = 'Dataset 5myts'
    file_name = 'scatterplot-5myts.png'

    x = np.array(input)[:, 2] - np.mean(np.array(input)[:, 5:], axis=1)  # x: front sensor - mean(rear sensors)
    y = np.array(output).flatten()                                       # y: speed
    x_label = 'sensing'
    y_label = 'control'

    scatter_plot(x, y, x_label, y_label, 'models/distributed/images/', title, file_name)

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

    outputs = []
    n_train = len(train_dataset)
    n_test = len(test_dataset)

    for _ in tqdm(range(epochs)):
        avg_loss = 0.0
        epoch_outputs = []

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

            avg_loss += float(loss) * batch_size
            optimizer.step()
            optimizer.zero_grad()

        training_loss.append(avg_loss / n_train)

        with torch.no_grad():
            test_losses = []
            outputs = []
            for inputs, labels in test_minibatch:
                t_output = net(inputs)
                # padded is used when in the simulations are different number of thymio
                if padded:
                    losses = []
                    for out, label in zip(inputs, labels):
                        label = unmask(label)
                        loss = criterion(out, label)
                        losses.append(loss)
                    loss = torch.mean(torch.stack(losses))
                else:
                    loss = criterion(t_output, labels)

                test_losses.append(float(loss))
                outputs.append(t_output)
            testing_loss.append(sum(test_losses) / n_test)
            epoch_outputs.append(outputs)
        print(avg_loss / n_train)

    return training_loss, testing_loss, outputs


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

    if len(sys.argv) > 1:
        if sys.argv[2] == 'load':
            command_dis = False
        if sys.argv[4] == 'save':
            save_cmd = True

    if command_dis:
        d_net = DistributedNet(x_train.shape[1])
        d_training_loss, d_testing_loss = [], []

        training_loss, testing_loss, outputs = train_net(epochs=20, net=d_net, train_dataset=d_training_set,
                                                         test_dataset=d_test_set, batch_size=100, learning_rate=0.001,
                                                         training_loss=d_training_loss, testing_loss=d_testing_loss)

        print('training_loss %s, testing_loss %s.' % (training_loss, testing_loss))

        img_dir = out_dir + 'images/'
        check_dir(img_dir)

        # Plot train and test losses
        title = 'Loss %s' % model
        file_name = 'loss-%s.png' % model
        plot_train_test_losses(training_loss, testing_loss, img_dir, title, file_name)

        file_name = 'loss-rescaled%s.png' % model
        plot_train_test_losses(training_loss, testing_loss, img_dir, title, file_name, scale=min(testing_loss) * 10)

        # Plot scatter-plot that compares the groundtruth to the prediction
        title = 'groundtruth vs prediction %s' % model
        file_name = 'gt-prediction-%s.png' % model
        x = torch.flatten(y_test).tolist()
        y = torch.flatten(torch.cat(outputs, dim=0)).tolist()
        x_label = 'groundtruth'
        y_label = 'prediction'
        scatter_plot(x, y, x_label, y_label, img_dir, title, file_name)

        # Plot prediction histogram
        title = 'Histogram Predictions %s' % model
        file_name = 'histogram-predictions%s.png' % model
        plot_prediction_histogram(y, 'prediction', img_dir, title, file_name)

        # Plot groundtruth histogram
        title = 'Histogram Groundtruth %s' % model
        file_name = 'histogram-groundtruth%s.png' % model
        plot_prediction_histogram(x, 'groundtruth', img_dir, title, file_name)

        # Plot sensing histogram
        title = 'Histogram Sensing %s' % model
        file_name = 'histogram-sensing%s.png' % model
        sensing = list(map(list, zip(*x_train.tolist())))
        x = [sensing[0], sensing[1], sensing[2], sensing[3], sensing[4], sensing[5], sensing[6]]
        label = ['prox_sens_0', 'prox_sens_1', 'prox_sens_2', 'prox_sens_3',
                 'prox_sens_4', 'prox_sens_5', 'prox_sens_6']
        plot_prediction_histogram(x, 'sensing', img_dir, title, file_name, label)

        if save_cmd:
            torch.save(d_net, '%s/%s' % (out_dir, model))
    else:
        u = 1
        d_net = torch.load('%s/%s' % (out_dir, model))


if __name__ == '__main__':

    out_dir = 'models/distributed/'
    check_dir(out_dir)
    file = os.path.join(out_dir, 'dataset_split.npy')

    runs_dir = 'out/5myts/'
    model = 'net1'

    try:
        main(file, runs_dir, out_dir, model)
    except:
        raise

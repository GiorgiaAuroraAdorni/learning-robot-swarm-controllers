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
from utils import check_dir, plot_losses, my_scatterplot, my_histogram


def from_dataset_to_tensors(runs_dir, train_indices, validation_indices, test_indices):
    """
    :param runs_dir: directory containing the simulations
    :param train_indices
    :param validation_indices
    :param test_indices
    :return: tensors with input and output of the network for both train and test set.
    """
    train_sample = []
    valid_sample = []
    test_sample = []
    train_target = []
    valid_target = []
    test_target = []

    input = []
    output = []

    indices = []
    for file_name in os.listdir(runs_dir):
        if not file_name.endswith('.pkl'):
            continue

        pickle_file = os.path.join(runs_dir, file_name)
        run = pd.read_pickle(pickle_file)

        i = int(re.findall('\d+', file_name)[0])
        indices.append(i)

        if i in train_indices:
            input = train_sample
            output = train_target
        elif i in validation_indices:
            input = valid_sample
            output = valid_target
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
    y = np.array(output).flatten()  # y: speed
    x_label = 'sensing'
    y_label = 'control'

    my_scatterplot(x, y, x_label, y_label, 'models/distributed/images/dataset', title, file_name)

    # Generate the tensors
    train_sample_tensor = torch.tensor(train_sample)
    valid_sample_tensor = torch.tensor(valid_sample)
    test_sample_tensor = torch.tensor(test_sample)
    train_target_tensor = torch.tensor(train_target)
    valid_target_tensor = torch.tensor(valid_target)
    test_target_tensor = torch.tensor(test_target)

    return train_sample_tensor, valid_sample_tensor, test_sample_tensor, train_target_tensor, valid_target_tensor, \
           test_target_tensor


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
              valid_dataset: data.TensorDataset,
              test_dataset: data.TensorDataset,
              net: torch.nn.Module,
              batch_size: int = 100,
              learning_rate: float = 0.01,
              training_loss: Optional[List[float]] = None,
              validation_loss: Optional[List[float]] = None,
              testing_loss: Optional[List[float]] = None,
              criterion=torch.nn.MSELoss(),
              padded=False,
              ) -> Tuple[List[float], List[float], List[float], Optional[Tuple[torch.Tensor, ...]]]:
    """
    :param epochs:
    :param train_dataset:
    :param valid_dataset:
    :param test_dataset:
    :param net:
    :param batch_size:
    :param learning_rate:
    :param training_loss:
    :param validation_loss:
    :param testing_loss:
    :param criterion:
    :param padded:
    :return:

    """

    train_minibatch = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_minibatch = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_minibatch = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    optimizer.zero_grad()

    if training_loss is None:
        training_loss = []

    if validation_loss is None:
        validation_loss = []

    if testing_loss is None:
        testing_loss = []

    outputs = []
    n_train = len(train_dataset)
    n_valid = len(valid_dataset)
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
            valid_losses = []
            outputs = []
            for inputs, labels in valid_minibatch:
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

                valid_losses.append(float(loss))
                outputs.append(t_output)
            validation_loss.append(sum(valid_losses) / n_valid)
            epoch_outputs.append(outputs)
        print(avg_loss / n_train)

    return training_loss, validation_loss, testing_loss, outputs


def network_plots(model, outputs, training_loss, validation_loss, x_train, y_valid):
    """

    :param model:
    :param outputs:
    :param training_loss:
    :param validation_loss:
    :param x_train:
    :param y_valid:
    """
    img_dir = out_dir + 'images/'
    check_dir(img_dir)

    # Plot train and test losses
    title = 'Loss %s' % model
    file_name = 'loss-%s.png' % model
    plot_losses(training_loss, validation_loss, img_dir, title, file_name)

    file_name = 'loss-rescaled-%s.png' % model
    plot_losses(training_loss, validation_loss, img_dir, title, file_name, scale=min(validation_loss) * 10)

    # Plot scatter-plot that compares the groundtruth to the prediction
    title = 'groundtruth vs prediction %s' % model
    file_name = 'gt-prediction-%s.png' % model

    x = torch.flatten(y_valid).tolist()
    y = torch.flatten(torch.cat(outputs, dim=0)).tolist()
    x_label = 'groundtruth'
    y_label = 'prediction'
    my_scatterplot(x, y, x_label, y_label, img_dir, title, file_name)

    # Plot prediction histogram
    title = 'Histogram Predictions Validation Set %s' % model
    file_name = 'histogram-predictions-validation-%s.png' % model
    my_histogram(y, 'prediction', img_dir, title, file_name)

    # Plot groundtruth histogram
    title = 'Histogram Groundtruth Validation Set %s' % model
    file_name = 'histogram-groundtruth-validation-%s.png' % model
    my_histogram(x, 'groundtruth', img_dir, title, file_name)

    # Plot sensing histogram
    title = 'Histogram Sensing Validation Set%s' % model
    file_name = 'histogram-sensing-validation-%s.png' % model

    sensing = list(map(list, zip(*x_train.tolist())))
    x = [sensing[0], sensing[1], sensing[2], sensing[3], sensing[4], sensing[5], sensing[6]]
    label = ['prox_sens_0', 'prox_sens_1', 'prox_sens_2', 'prox_sens_3',
             'prox_sens_4', 'prox_sens_5', 'prox_sens_6']
    my_histogram(x, 'sensing', img_dir, title, file_name, label)


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
    n_train = 600
    n_validation = 200
    train_indices, validation_indices, test_indices = dataset[:n_train], \
                                                      dataset[n_train:n_train + n_validation], \
                                                      dataset[n_train + n_validation:]

    # Split the dataset also defining input and output, using the indices
    x_train, x_valid, x_test, y_train, y_valid, y_test = from_dataset_to_tensors(runs_dir,
                                                                                 train_indices,
                                                                                 validation_indices,
                                                                                 test_indices)

    d_train_set = TensorDataset(x_train, y_train)
    d_valid_set = TensorDataset(x_valid, y_valid)
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
        d_training_loss, d_validation_loss, d_testing_loss = [], [], []

        training_loss, validation_loss, testing_loss, outputs = train_net(epochs=20,
                                                                          train_dataset=d_train_set,
                                                                          valid_dataset=d_valid_set,
                                                                          test_dataset=d_test_set,
                                                                          net=d_net,
                                                                          training_loss=d_training_loss,
                                                                          validation_loss=d_validation_loss,
                                                                          testing_loss=d_testing_loss)

        print('training_loss %s, validation_loss %s.' % (training_loss, validation_loss))

        network_plots(model, outputs, training_loss, validation_loss, x_train, y_valid)

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

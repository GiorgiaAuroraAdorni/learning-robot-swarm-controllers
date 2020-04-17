import os
import re
from typing import List, Tuple, Optional, AnyStr

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
from tqdm import tqdm

from networks.distributed_network import DistributedNet
from utils import check_dir, plot_losses, my_scatterplot, my_histogram, dataset_split


def from_indices_to_dataset(runs_dir, train_indices, validation_indices, test_indices):
    """
    :param runs_dir: directory containing the simulations
    :param train_indices
    :param validation_indices
    :param test_indices
    :return: train_sample, valid_sample, test_sample, train_target, valid_target, test_target, input, output
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
                # they are normalised so that the average is around 1 or a constant (e.g. for all (dividing by 1000))
                sample = myt['prox_values'].copy()
                normalised_sample = np.divide(np.array(sample), 1000).tolist()
                input.append(normalised_sample)

                # The output is the speed of the wheels (which we assume equals left and right) [array of 1 float]
                # There is no need to normalize the outputs.
                speed = myt['motor_left_target']
                output.append([speed])

    return train_sample, valid_sample, test_sample, train_target, valid_target, test_target, input, output


def from_dataset_to_tensors(test_sample, test_target, train_sample, train_target, valid_sample, valid_target):
    """

    :param test_sample:
    :param test_target:
    :param train_sample:
    :param train_target:
    :param valid_sample:
    :param valid_target:
    :return d_test_set, d_train_set, d_valid_set, x_train, y_valid:
    """
    x_train = torch.tensor(train_sample)
    x_valid = torch.tensor(valid_sample)
    x_test = torch.tensor(test_sample)

    y_train = torch.tensor(train_target)
    y_valid = torch.tensor(valid_target)
    y_test = torch.tensor(test_target)

    d_train_set = TensorDataset(x_train, y_train)
    d_valid_set = TensorDataset(x_valid, y_valid)
    d_test_set = TensorDataset(x_test, y_test)

    return d_test_set, d_train_set, d_valid_set, x_train, y_valid


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


def train_net(file: AnyStr,
              epochs: int,
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
              padded=False
              ) -> Tuple[List[float], List[float], List[float]]:
    """
    :param file:
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
    :return training_loss, validation_loss, testing_loss:

    """

    train_minibatch = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_minibatch = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_minibatch = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    np.save(file, [train_minibatch, valid_minibatch, test_minibatch])

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    if training_loss is None:
        training_loss = []

    if validation_loss is None:
        validation_loss = []

    if testing_loss is None:
        testing_loss = []

    n_train = len(train_dataset)
    n_valid = len(valid_dataset)
    n_test = len(test_dataset)

    for _ in tqdm(range(epochs)):
        avg_loss = 0.0

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
        print(avg_loss / n_train)

        validate_net(n_valid, net, valid_minibatch, validation_loss, padded, criterion)

    return training_loss, validation_loss, testing_loss


def validate_net(n_valid, net, valid_minibatch, validation_loss, padded=False, criterion=torch.nn.MSELoss()):
    """
    :param n_valid:
    :param net:
    :param valid_minibatch:
    :param validation_loss:
    :param padded:
    :param criterion:
    :return outputs:
    """
    valid_losses = []
    outputs = []

    with torch.no_grad():
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

    return outputs


def network_plots(dataset, outputs, training_loss, validation_loss, x_train, y_valid, input, output):
    """

    :param dataset:
    :param outputs:
    :param training_loss:
    :param validation_loss:
    :param x_train:
    :param y_valid:
    :param input:
    :param output:
    """
    img_dir = '%s/images/' % out_dir
    check_dir(img_dir)

    #  Generate a scatter plot to check the conformity of the dataset
    title = 'Dataset %s' % dataset
    file_name = 'dataset-scatterplot-%s.png' % dataset

    x = np.array(input)[:, 2] - np.mean(np.array(input)[:, 5:], axis=1)  # x: front sensor - mean(rear sensors)
    y = np.array(output).flatten()  # y: speed
    x_label = 'sensing'
    y_label = 'control'

    my_scatterplot(x, y, x_label, y_label, '%s/dataset' % img_dir, title, file_name)

    # Plot train and test losses
    title = 'Loss %s' % dataset
    file_name = 'loss-%s.png' % dataset
    plot_losses(training_loss, validation_loss, img_dir, title, file_name)

    file_name = 'loss-rescaled-%s.png' % dataset
    plot_losses(training_loss, validation_loss, img_dir, title, file_name, scale=min(validation_loss) * 10)

    # Plot scatter-plot that compares the groundtruth to the prediction
    title = 'groundtruth vs prediction %s' % dataset
    file_name = 'gt-prediction-%s.png' % dataset

    x = torch.flatten(y_valid).tolist()
    y = torch.flatten(torch.cat(outputs, dim=0)).tolist()
    x_label = 'groundtruth'
    y_label = 'prediction'
    my_scatterplot(x, y, x_label, y_label, img_dir, title, file_name)

    # Plot prediction histogram
    title = 'Histogram Predictions Validation Set %s' % dataset
    file_name = 'histogram-predictions-validation-%s.png' % dataset
    my_histogram(y, 'prediction', img_dir, title, file_name)

    # Plot groundtruth histogram
    title = 'Histogram Groundtruth Validation Set %s' % dataset
    file_name = 'histogram-groundtruth-validation-%s.png' % dataset
    my_histogram(x, 'groundtruth', img_dir, title, file_name)

    # Plot sensing histogram
    title = 'Histogram Sensing Validation Set%s' % dataset
    file_name = 'histogram-sensing-validation-%s.png' % dataset

    sensing = list(map(list, zip(*x_train.tolist())))
    x = [sensing[0], sensing[1], sensing[2], sensing[3], sensing[4], sensing[5], sensing[6]]
    label = ['prox_sens_0', 'prox_sens_1', 'prox_sens_2', 'prox_sens_3',
             'prox_sens_4', 'prox_sens_5', 'prox_sens_6']
    my_histogram(x, 'sensing', img_dir, title, file_name, label)


def main(file, runs_dir, out_dir, model, ds, train):
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
    train_sample, valid_sample, test_sample, \
    train_target, valid_target, test_target, \
    input, output = from_indices_to_dataset(runs_dir, train_indices, validation_indices, test_indices)

    # Generate the tensors
    d_test_set, d_train_set, d_valid_set, x_train, y_valid = from_dataset_to_tensors(test_sample, test_target,
                                                                                     train_sample, train_target,
                                                                                     valid_sample, valid_target)

    file_losses = os.path.join(out_dir, 'losses.npy')
    file_minibatch = os.path.join(out_dir, 'minibatch.npy')

    if train:
        d_net = DistributedNet(x_train.shape[1])
        d_training_loss, d_validation_loss, d_testing_loss = [], [], []

        training_loss, validation_loss, testing_loss = train_net(file=file_minibatch,
                                                                 epochs=20,
                                                                 train_dataset=d_train_set,
                                                                 valid_dataset=d_valid_set,
                                                                 test_dataset=d_test_set,
                                                                 net=d_net,
                                                                 training_loss=d_training_loss,
                                                                 validation_loss=d_validation_loss,
                                                                 testing_loss=d_testing_loss)

        np.save(file_losses, [training_loss, validation_loss, testing_loss])

        torch.save(d_net, '%s/%s' % (out_dir, model))
    else:
        d_net = torch.load('%s/%s' % (out_dir, model))

        # Load the metrics
        training_loss, validation_loss, testing_loss = np.load(file_losses, allow_pickle=True)
        train_minibatch, valid_minibatch, test_minibatch = np.load(file_minibatch, allow_pickle=True)

        outputs = validate_net(len(d_valid_set), d_net, valid_minibatch, validation_loss.copy())

        network_plots(ds, outputs, training_loss, validation_loss, x_train, y_valid, input, output)


if __name__ == '__main__':
    model = 'net1'
    dataset = '5myts'

    runs_dir = 'out/%s/' % dataset
    out_dir = 'models/distributed/%s' % model
    check_dir(out_dir)
    check_dir(runs_dir)

    file = os.path.join(out_dir, 'dataset_split.npy')

    try:
        main(file, runs_dir, out_dir, model, dataset, train=False)
    except:
        raise

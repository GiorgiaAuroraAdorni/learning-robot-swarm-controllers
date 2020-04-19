import os
import os
import re
from typing import List, Tuple, Optional, AnyStr

import numpy as np
import pandas as pd
import torch
from networks.distributed_network import DistributedNet
from pid import PID
from torch.utils import data
from torch.utils.data import TensorDataset
from tqdm import tqdm
from utils import check_dir, plot_losses, my_scatterplot, my_histogram, plot_regressor, plot_response


def from_indices_to_dataset(runs_dir, train_indices, validation_indices, test_indices):
    """
    :param runs_dir: directory containing the simulations
    :param train_indices
    :param validation_indices
    :param test_indices
    :return: train_sample, valid_sample, test_sample, train_target, valid_target, test_target, input_, output_
    """
    train_sample = []
    valid_sample = []
    test_sample = []
    train_target = []
    valid_target = []
    test_target = []

    input_ = []
    output_ = []

    indices = []
    for file_name in os.listdir(runs_dir):
        if not file_name.endswith('.pkl'):
            continue

        pickle_file = os.path.join(runs_dir, file_name)
        run = pd.read_pickle(pickle_file)

        i = int(re.findall('\d+', file_name)[0])
        indices.append(i)

        if i in train_indices:
            input_ = train_sample
            output_ = train_target
        elif i in validation_indices:
            input_ = valid_sample
            output_ = valid_target
        elif i in test_indices:
            input_ = test_sample
            output_ = test_target

        for step in run:
            for myt in step:
                # The input is the prox_values, that are the response values of ​​the sensors [array of 7 floats]
                # they are normalised so that the average is around 1 or a constant (e.g. for all (dividing by 1000))
                sample = myt['prox_values'].copy()
                normalised_sample = np.divide(np.array(sample), 1000).tolist()
                input_.append(normalised_sample)

                # The output is the speed of the wheels (which we assume equals left and right) [array of 1 float]
                # There is no need to normalize the outputs.
                speed = myt['motor_left_target']
                output_.append([speed])

    return train_sample, valid_sample, test_sample, train_target, valid_target, test_target, input_, output_


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
        validation_loss.append(sum(valid_losses) / n_valid)


def network_plots(runs_img, model_dir, dataset, prediction, training_loss, validation_loss, x_train, y_valid,
                  sensing, groundtruth):
    """

    :param runs_img
    :param model_dir
    :param dataset:
    :param prediction:
    :param training_loss:
    :param validation_loss:
    :param x_train:
    :param y_valid:
    :param sensing:
    :param groundtruth:
    """
    model_img = '%s/images/' % model_dir
    check_dir(model_img)

    #  Generate a scatter plot to check the conformity of the dataset
    title = 'Dataset %s' % dataset
    file_name = 'dataset-scatterplot-%s.pdf' % dataset

    x = np.array(sensing)[:, 2] - np.mean(np.array(sensing)[:, 5:], axis=1)  # x: front sensor - mean(rear sensors)
    y = np.array(groundtruth).flatten()  # y: speed
    x_label = 'sensing'
    y_label = 'control'

    my_scatterplot(x, y, x_label, y_label, runs_img, title, file_name)

    # Plot train and validation losses
    title = 'Loss %s' % dataset
    file_name = 'loss-%s.pdf' % dataset
    plot_losses(training_loss, validation_loss, model_img, title, file_name)

    file_name = 'loss-rescaled-%s.pdf' % dataset
    plot_losses(training_loss, validation_loss, model_img, title, file_name, scale=min(validation_loss) * 10)

    # Plot groundtruth histogram
    title = 'Groundtruth Validation Set %s' % dataset
    file_name = 'histogram-groundtruth-validation-%s.pdf' % dataset
    my_histogram(y, 'groundtruth', model_img, title, file_name)

    # Plot prediction histogram
    title = 'Prediction Validation Set %s' % dataset
    file_name = 'histogram-prediction-validation-%s.pdf' % dataset
    prediction = torch.flatten(prediction).tolist()
    my_histogram(prediction, 'prediction', model_img, title, file_name)

    # Plot sensing histogram
    title = 'Sensing Validation Set%s' % dataset
    file_name = 'histogram-sensing-validation-%s.pdf' % dataset

    x = [x_train[0], x_train[1], x_train[2], x_train[3], x_train[4], x_train[5], x_train[6]]
    label = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']
    my_histogram(x, 'sensing', model_img, title, file_name, label)

    # Plot R^2 of the regressor between prediction and ground truth on the validation set
    # title = 'Regression learned %s vs %s' % dataset
    title = 'Regression learned controller vs %s' % dataset
    file_name = 'regression-learned-vs-%s.pdf' % dataset

    y_valid = np.array(y_valid).flatten()
    x_label = 'groundtruth'
    y_label = 'prediction'
    plot_regressor(y_valid, prediction, x_label, y_label, model_img, title, file_name)


def controller_plots(model_dir, ds, ds_eval, prediction, groundtruth):
    """

    :param model_dir
    :param ds:
    :param ds_eval:
    :param prediction:
    :param groundtruth:
    """
    model_img = '%s/images/' % model_dir
    check_dir(model_img)

    # Plot R^2 of the regressor between prediction and ground truth
    title = 'Regression %s vs %s' % (ds_eval, ds)
    file_name = 'regression-%svs%s.pdf' % (ds_eval, ds)

    groundtruth = np.array(groundtruth).flatten()
    prediction = np.array(prediction).flatten()
    x_label = 'groundtruth'
    y_label = 'prediction'
    plot_regressor(groundtruth, prediction, x_label, y_label, model_img, title, file_name)


def neighbors_distance(sensing):
    """
    :param sensing
    Check if there is a robot ahead using the infrared sensor 2 (front-front).
    Check if there is a robot ahead using the infrared sensor 5 (back-left) and 6 (back-right).
    :return back, front: response values of the rear and front sensors
    """
    front = sensing[2]
    back = np.mean(np.array([sensing[5], sensing[6]]))

    return back, front


def compute_difference(sensing):
    """
    :param sensing
    :return out: the difference between the response value of front and the rear sensor
    """
    back, front = neighbors_distance(sensing)

    # Apply a small correction to the distance measured by the rear sensors: the front sensor used is at a
    # different x coordinate from the point to which the rear sensor of the robot that follows points. this is
    # because of the curved shape of the face of the Thymio
    delta_x = 7.41150769
    x = 7.95

    # Maximum possible response values
    delta_x_m = 4505 * delta_x / 14
    x_m = 4505 * x / 14

    correction = x_m - delta_x_m

    out = front - correction - back

    return out


def distributed_controller(sensing, dt=0.1):
    """

    :param sensing
    :param dt: timestep
    :return speed

    """
    p_distributed_controller = PID(-0.01, 0, 0, max_out=16.6, min_out=-16.6)
    speed = p_distributed_controller.step(compute_difference(sensing), dt)

    return speed


def compare_net_to_manual_controller(model_dir, ds, ds_eval, sensing, groundtruth):
    """

    :param model_dir:
    :param ds:
    :param ds_eval:
    :param sensing:
    :param groundtruth:
    """
    controller_predictions = []

    for sample in sensing:
        # Rescale the values of the sensor
        sample = np.multiply(np.array(sample), 1000).tolist()
        control = distributed_controller(sample)
        controller_predictions.append(control)

    controller_plots(model_dir, ds, ds_eval, controller_predictions, groundtruth)


def generate_sensing():
    """

    :return sensing
    """
    x = np.arange(4500)
    s = np.zeros(x.shape[0])

    sensing = np.stack([s, s, np.divide(x, 1000), s, s, s, s], axis=1)

    return sensing


def evaluate_net(model_dir, model, d_net):
    """

    :param model_dir:
    :param model:
    :param d_net:
    """
    sensing = generate_sensing()
    d_sensing = torch.FloatTensor(sensing)

    predictions = d_net(d_sensing)

    model_img = '%s/images/' % model_dir
    check_dir(model_img)

    title = 'Response %s' % model
    file_name = 'response-%s.pdf' % model

    # Plot the output of the network given the input ([0, 0, x, 0, 0, 0, 0]) for x in [0, 4500]
    plot_response(sensing, predictions, model_img, title, file_name)


def main(file, runs_dir, runs_img, model_dir, model, ds, ds_eval, train):
    """
    :param file: file containing the defined indices for the split
    :param runs_dir: directory containing the simulations
    :param runs_img: directory containing the images related to the dataset
    :param model_dir: directory containing the network data
    :param model:
    :param ds
    :param ds_eval
    :param train
    """
    # Uncomment the following line to generate a new dataset split
    # dataset_split(file)

    # Load the indices
    dataset = np.load(file)
    n_train = 600
    n_validation = 800
    train_indices, validation_indices, test_indices = dataset[:n_train], dataset[n_train:n_validation], dataset[
                                                                                                        n_validation:]

    # Split the dataset also defining input and output, using the indices
    train_sample, valid_sample, test_sample, \
    train_target, valid_target, test_target, \
    sensing, groundtruth = from_indices_to_dataset(runs_dir, train_indices, validation_indices, test_indices)

    # Generate the tensors
    d_test_set, d_train_set, d_valid_set, x_train, y_valid = from_dataset_to_tensors(test_sample, test_target,
                                                                                     train_sample, train_target,
                                                                                     valid_sample, valid_target)

    file_losses = os.path.join(model_dir, 'losses.npy')
    file_minibatch = os.path.join(model_dir, 'minibatch.npy')

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

        torch.save(d_net, '%s/%s' % (model_dir, model))
    else:
        d_net = torch.load('%s/%s' % (model_dir, model))

        # Load the metrics
        training_loss, validation_loss, testing_loss = np.load(file_losses, allow_pickle=True)

        prediction = d_net(torch.FloatTensor(valid_sample))

        network_plots(runs_img, model_dir, ds, prediction, training_loss, validation_loss, train_sample, valid_target,
                      sensing, groundtruth)

        compare_net_to_manual_controller(model_dir, ds, ds_eval, sensing, groundtruth)

        evaluate_net(model_dir, model, d_net)


if __name__ == '__main__':
    model = 'net1'
    myt_quantity = 5

    dataset_net = '%dmyts-%s' % (myt_quantity, 'omniscient')
    dataset_eval = '%dmyts-%s' % (myt_quantity, 'distributed')

    runs_dir = os.path.join('datasets/', dataset_net)
    runs_img = '%s/images/' % runs_dir
    model_dir = 'models/distributed/%s' % model

    check_dir(runs_dir)
    check_dir(runs_img)
    check_dir(model_dir)

    file = os.path.join('models/distributed/', 'dataset_split.npy')

    main(file, runs_dir, runs_img, model_dir, model, dataset_net, dataset_eval, train=False)

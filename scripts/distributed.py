import os
from typing import List, Tuple, Optional, AnyStr

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
from tqdm import tqdm

from controllers import distributed_controllers
from generate_simulation_data import GenerateSimulationData as g
from my_plots import plot_regressor, plot_response, my_histogram, plot_losses
from networks.distributed_network import DistributedNet
from utils import check_dir, extract_input_output, dataset_split


class ThymioState:
    def __init__(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)


def from_indices_to_dataset(runs_dir, train_indices, validation_indices, test_indices, net_input):
    """
    :param runs_dir: directory containing the simulations
    :param train_indices
    :param validation_indices
    :param test_indices
    :param net_input
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

    pickle_file = os.path.join(runs_dir, 'simulation.pkl')
    runs = pd.read_pickle(pickle_file)

    for i, run in enumerate(runs):

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

        extract_input_output(run, input_, output_, net_input, 'motor_left_target')

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

        validate_net(n_valid, net, valid_minibatch, validation_loss, batch_size, padded, criterion)

    return training_loss, validation_loss, testing_loss


def validate_net(n_valid, net, valid_minibatch, validation_loss, batch_size, padded=False,
                 criterion=torch.nn.MSELoss()):
    """
    :param n_valid:
    :param net:
    :param valid_minibatch:
    :param validation_loss:
    :param batch_size
    :param padded:
    :param criterion:
    :return outputs:
    """
    avg_loss = 0.0

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
            avg_loss += float(loss) * batch_size
        validation_loss.append(avg_loss / n_valid)


def network_plots(model_img, dataset, model, net_input, prediction, training_loss, validation_loss, x_train, y_valid,
                  groundtruth):
    """
    :param model_img
    :param dataset:
    :param model
    :param net_input
    :param prediction:
    :param training_loss:
    :param validation_loss:
    :param x_train:
    :param y_valid:
    :param groundtruth:
    """
    y = np.array(groundtruth).flatten()  # y: speed

    # Plot train and validation losses
    title = 'Loss %s' % model
    file_name = 'loss-%s' % model
    plot_losses(training_loss, validation_loss, model_img, title, file_name)

    # file_name = 'loss-rescaled-%s' % dataset
    # plot_losses(training_loss, validation_loss, model_img, title, file_name, scale=min(validation_loss) * 10)

    # Plot groundtruth histogram
    title = 'Groundtruth Validation Set %s' % model
    file_name = 'histogram-groundtruth-validation-%s' % model
    my_histogram(y, 'groundtruth', model_img, title, file_name)

    # Plot prediction histogram
    title = 'Prediction Validation Set %s' % model
    file_name = 'histogram-prediction-validation-%s' % model
    prediction = torch.flatten(prediction).tolist()
    my_histogram(prediction, 'prediction', model_img, title, file_name)

    if not net_input == 'all_sensors':
        # Plot sensing histogram
        title = 'Sensing Validation Set%s' % model
        file_name = 'histogram-sensing-validation-%s' % model

        x = [x_train[0], x_train[1], x_train[2], x_train[3], x_train[4], x_train[5], x_train[6]]
        label = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']
        my_histogram(x, 'sensing (%s)' % net_input, model_img, title, file_name, label)

    # Evaluate prediction of the learned controller to the omniscient groundtruth
    # Plot R^2 of the regressor between prediction and ground truth on the validation set
    # title = 'Regression learned %s vs %s' % dataset
    title = 'Regression %s vs %s' % (model, dataset)
    file_name = 'regression-%s-vs-%s' % (model, dataset)

    y_valid = np.array(y_valid).flatten()
    x_label = 'groundtruth'
    y_label = 'prediction'
    plot_regressor(y_valid, prediction, x_label, y_label, model_img, title, file_name)


def controller_plots(model_dir, ds, ds_eval, groundtruth, prediction):
    """

    :param model_dir
    :param ds:
    :param ds_eval:
    :param groundtruth:
    :param prediction:
    """
    model_img = '%s/images/' % model_dir
    check_dir(model_img)

    # Plot R^2 of the regressor between prediction and ground truth
    title = 'Regression %s vs %s' % (ds_eval, ds)
    file_name = 'regression-%svs%s' % (ds_eval, ds)

    groundtruth = np.array(groundtruth).flatten()
    prediction = np.array(prediction).flatten()
    x_label = 'groundtruth'
    y_label = 'prediction'
    plot_regressor(groundtruth, prediction, x_label, y_label, model_img, title, file_name)


def evaluate_controller(model_dir, ds, ds_eval, groundtruth, sensing, net_input):
    """

    :param model_dir:
    :param ds:
    :param ds_eval:
    :param groundtruth:
    :param sensing: used to obtain the prediction
    :param net_input
    """
    controller_predictions = []
    controller = distributed_controllers.ManualController(net_input=net_input)

    for sample in sensing:
        # Rescale the values of the sensor
        sample = np.multiply(np.array(sample), 1000).tolist()

        if net_input == 'prox_comm':
            prox_comm = {'sender': {'intensities': sample}}
            prox_values = None
        elif net_input == 'prox_values':
            prox_values = sample
            prox_comm = None
        elif net_input == 'all_sensors':
            prox_values = sample[:7]
            prox_comm = {'sender': {'intensities': sample[7:]}}
        else:
            raise AttributeError("Invalid value for net_input")

        state_dict = {'initial_position': (0, 0), 'goal_position': (10, 0), 'prox_values': prox_values,
                      'prox_comm': prox_comm}

        state = ThymioState(state_dict)

        control = controller.perform_control(state, dt=0.1)

        controller_predictions.append(control)

    controller_plots(model_dir, ds, ds_eval, groundtruth, controller_predictions)


def generate_sensing():
    """

    :return x, s: x is the configuration, s is the zeros array
    """
    x = np.arange(4500)
    s = np.zeros(x.shape[0])

    return x, s


def evaluate_net(model_img, model, net, net_input, net_title, sensing, index, x_label):
    """

    :param model_img:
    :param model:
    :param net:
    :param net_input
    :param net_title
    :param sensing:
    :param index:
    :param x_label:
    """
    controller_predictions = []
    controller = distributed_controllers.LearnedController(net=net, net_input=net_input)

    for sample in sensing:
        # Rescale the values of the sensor
        sample = np.multiply(np.array(sample), 1000).tolist()

        if net_input == 'prox_comm':
            prox_comm = {'sender': {'intensities': sample}}
            prox_values = None
        elif net_input == 'prox_values':
            prox_values = sample
            prox_comm = None
        else:
            raise AttributeError('Input not found')

        state_dict = {'initial_position': (0, 0), 'goal_position': (10, 0), 'prox_values': prox_values,
                      'prox_comm': prox_comm}

        state = ThymioState(state_dict)

        control = controller.perform_control(state, dt=0.1)

        controller_predictions.append(control)

    title = 'Response %s - %s' % (model, net_title)
    file_name = 'response-%s-%s' % (model, net_title)

    # Plot the output of the network
    plot_response(sensing, controller_predictions, x_label, model_img, title, file_name, index)


def test_controller_given_init_positions(model_img, net, model, net_input, avg_gap):
    """

    :param model_img:
    :param net:
    :param model:
    :param net_input
    :param avg_gap
    """
    myt_quantity = 3

    def controller_factory(**kwargs):
        return distributed_controllers.LearnedController(net=net, net_input=net_input, **kwargs)

    # FIXME do not use simulation
    world, myts = g.setup(controller_factory, myt_quantity)

    simulations = 17 * 10

    range = 2 * avg_gap

    x = np.linspace(0, range, num=simulations)
    control_predictions = []

    for simulation in tqdm(x):
        g.init_positions(myts, net_input, avg_gap, variate_pose=True, x=simulation)

        world.step(dt=0.1)
        # myts[1].learned_controller()
        control = myts[1].motor_left_target
        control_predictions.append(control)

    title = 'Response %s by varying init position' % model
    file_name = 'response-%s-varying_init_position' % model

    # Plot the output of the network
    plot_response(x, control_predictions, 'init avg gap', model_img, title, file_name)


def run_distributed(file, runs_dir, model_dir, model_img, model, ds, ds_eval, train, generate_split, plots,
                    net_input, avg_gap):
    """
    :param file: file containing the defined indices for the split
    :param runs_dir:
    :param model_dir: directory containing the network data
    :param model_img: directory containing the images related to network
    :param model
    :param ds
    :param ds_eval
    :param train
    :param generate_split
    :param plots
    :param net_input
    :param avg_gap
    """
    # Uncomment the following line to generate a new dataset split
    if generate_split:
        dataset_split(file)

    # Load the indices
    dataset = np.load(file)
    n_train = 600
    n_validation = 800
    train_indices, validation_indices, test_indices = dataset[:n_train], dataset[n_train:n_validation], \
                                                      dataset[n_validation:]

    # Split the dataset also defining input and output, using the indices
    train_sample, valid_sample, test_sample, \
    train_target, valid_target, test_target, \
    sensing, groundtruth = from_indices_to_dataset(runs_dir, train_indices, validation_indices, test_indices, net_input)

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
                                                                 epochs=50,
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

    if plots:
        # Load the metrics
        training_loss, validation_loss, testing_loss = np.load(file_losses, allow_pickle=True)

        prediction = d_net(torch.FloatTensor(valid_sample))

        network_plots(model_img, ds, model, net_input, prediction, training_loss, validation_loss, train_sample,
                      valid_target, groundtruth)

        # Evaluate prediction of the distributed controller with the omniscient groundtruth
        evaluate_controller(model_dir, ds, ds_eval, groundtruth, sensing, net_input)

        if not net_input == 'all_sensors':
            # Evaluate the learned controller by passing a specific input sensing configuration
            x, s = generate_sensing()
            sensing = np.stack([s, s, np.divide(x, 1000), s, s, s, s], axis=1)
            index = 2

            evaluate_net(model_img, model, d_net, net_input, 'net([0, 0, x, 0, 0, 0, 0])', sensing, index,
                         'center proximity sensor')

            index = -1
            sensing = np.stack([s, s, s, s, s, np.divide(x, 1000), np.divide(x, 1000)], axis=1)
            evaluate_net(model_img, model, d_net, net_input, 'net([0, 0, 0, 0, 0, x, x])', sensing, index,
                         'rear proximity sensors')

        # Evaluate the learned controller by passing a specific initial position configuration
        test_controller_given_init_positions(model_img, d_net, model, net_input, avg_gap)

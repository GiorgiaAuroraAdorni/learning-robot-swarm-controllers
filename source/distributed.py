import os
from typing import List, Tuple, Optional, AnyStr

import numpy as np
import torch
from torch.utils import data
from torch.utils.data import TensorDataset
from tqdm import tqdm

from controllers import distributed_controllers
from generate_simulation_data import GenerateSimulationData as g
from my_plots import plot_regressor, plot_response, my_histogram, plot_losses, plot_target_distribution
from networks.distributed_network import DistributedNet
import utils


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

    runs = utils.load_dataset(runs_dir, 'simulation.pkl')
    runs_sub = runs[['timestep', 'run', 'motor_left_target', 'prox_values', 'prox_comm', 'all_sensors']]

    input_, output_, _ = utils.extract_input_output(runs_sub, net_input, input_combination=False)

    train_runs = runs_sub[runs_sub['run'].isin(train_indices)].reset_index()
    valid_runs = runs_sub[runs_sub['run'].isin(validation_indices)].reset_index()
    test_runs = runs_sub[runs_sub['run'].isin(test_indices)].reset_index()

    train_sample, train_target, _ = utils.extract_input_output(train_runs, net_input, input_combination=False)
    valid_sample, valid_target, _ = utils.extract_input_output(valid_runs, net_input, input_combination=False)
    test_sample, test_target, _ = utils.extract_input_output(test_runs, net_input, input_combination=False)

    return train_sample, valid_sample, test_sample, \
           train_target[:, None], valid_target[:, None], test_target[:, None], \
           input_, output_[:, None]


def from_dataset_to_tensors(train_sample, train_target, valid_sample, valid_target, test_sample, test_target):
    """

    :param train_sample:
    :param train_target:
    :param valid_sample:
    :param valid_target:
    :param test_sample:
    :param test_target:
    :return t_d_test, t_d_train, t_d_valid:
    """
    x_train_tensor = torch.tensor(train_sample, dtype=torch.float32)
    x_valid_tensor = torch.tensor(valid_sample, dtype=torch.float32)
    x_test_tensor = torch.tensor(test_sample, dtype=torch.float32)

    y_train_tensor = torch.tensor(train_target, dtype=torch.float32)
    y_valid_tensor = torch.tensor(valid_target, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_target, dtype=torch.float32)

    t_d_train = TensorDataset(x_train_tensor, y_train_tensor)
    t_d_valid = TensorDataset(x_valid_tensor, y_valid_tensor)
    t_d_test = TensorDataset(x_test_tensor, y_test_tensor)

    return t_d_test, t_d_train, t_d_valid


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
              padded=False
              ) -> Tuple[List[float], List[float], List[float]]:
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
    :return training_loss, validation_loss, testing_loss:

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
        net.eval()
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


def network_plots(model_img, dataset, model, net_input, prediction, training_loss, validation_loss, x_train, y_valid):
    """
    :param model_img
    :param dataset:
    :param model
    :param net_input
    :param y_p:
    :param training_loss:
    :param validation_loss:
    :param x_train:
    :param y_valid:
    """
    y_p = prediction.squeeze().tolist()
    y_g = y_valid.squeeze().tolist()

    # Plot train and validation losses
    title = 'Loss %s' % model
    file_name = 'loss-%s' % model
    plot_losses(training_loss, validation_loss, model_img, title, file_name)

    # Plot target distribution
    title = 'Distribution Target Validation Set %s' % model
    file_name = 'distribution-target-validation-%s' %model
    plot_target_distribution(y_g, y_p, model_img, title, file_name)

    if not net_input == 'all_sensors':
        # Plot sensing histogram
        title = 'Sensing Validation Set %s' % model
        file_name = 'histogram-sensing-validation-%s' % model

        x = [x_train[0], x_train[1], x_train[2], x_train[3], x_train[4], x_train[5], x_train[6]]
        label = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']
        my_histogram(x, 'sensing (%s)' % net_input, model_img, title, file_name, label)

    # Evaluate prediction of the learned controller to the omniscient groundtruth
    # Plot R^2 of the regressor between prediction and ground truth on the validation set
    # title = 'Regression learned %s vs %s' % dataset
    title = 'Regression %s vs %s' % (model, dataset)
    file_name = 'regression-%s-vs-%s' % (model, dataset)

    x_label = 'groundtruth'
    y_label = 'prediction'
    plot_regressor(y_g, y_p, x_label, y_label, model_img, title, file_name)


def controller_plots(model_dir, ds, ds_eval, groundtruth, prediction):
    """

    :param model_dir
    :param ds:
    :param ds_eval:
    :param groundtruth:
    :param prediction:
    """
    model_img = '%s/images/' % model_dir
    utils.check_dir(model_img)

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
            prox_values = sample
            prox_comm = {'sender': {'intensities': sample}}
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
        utils.dataset_split(file)

    # Load the indices
    dataset = np.load(file)
    n_train = 600
    n_validation = 800
    train_indices, validation_indices, test_indices = dataset[:n_train], dataset[n_train:n_validation], \
                                                      dataset[n_validation:]

    # Split the dataset also defining input and output, using the indices
    x_train, x_valid, x_test, \
    y_train, y_valid, y_test, \
    sensing, groundtruth = from_indices_to_dataset(runs_dir, train_indices, validation_indices, test_indices, net_input)

    # Generate the tensors
    t_d_test, t_d_train, t_d_valid = from_dataset_to_tensors(x_train, y_train, x_valid, y_valid, x_test, y_test)

    file_losses = os.path.join(model_dir, 'losses.npy')

    if train:
        print('\nTraining %s…' % model)
        d_net = DistributedNet(x_train.shape[1])
        d_training_loss, d_validation_loss, d_testing_loss = [], [], []

        training_loss, validation_loss, testing_loss = train_net(epochs=50,
                                                                 train_dataset=t_d_train,
                                                                 valid_dataset=t_d_valid,
                                                                 test_dataset=t_d_test,
                                                                 net=d_net,
                                                                 training_loss=d_training_loss,
                                                                 validation_loss=d_validation_loss,
                                                                 testing_loss=d_testing_loss)

        np.save(file_losses, [training_loss, validation_loss, testing_loss])

        torch.save(d_net, '%s/%s' % (model_dir, model))
    else:
        d_net = torch.load('%s/%s' % (model_dir, model))

    if plots:
        print('\nGenerating plots for %s…' % model)
        # Load the metrics
        training_loss, validation_loss, testing_loss = np.load(file_losses, allow_pickle=True)

        prediction = d_net(torch.FloatTensor(x_valid))

        network_plots(model_img, ds, model, net_input, prediction, training_loss, validation_loss, x_train, y_valid)

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

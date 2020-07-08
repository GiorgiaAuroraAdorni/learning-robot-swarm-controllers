import numpy as np
import pandas as pd
import torch
import tqdm

from utils import utils
from controllers import controllers
from generate_simulation_data import GenerateSimulationData as g

class ThymioState:
    def __init__(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)


def network_plots(model_img, dataset, model, net_input, prediction, training_loss, validation_loss, x_train, y_valid,
                  communication):
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
    :param communication
    """
    y_p = prediction.squeeze().tolist()
    y_g = y_valid.squeeze().tolist()

    # Plot train and validation losses
    title = 'Loss %s' % model
    file_name = 'loss-%s' % model
    utils.plot_losses(training_loss, validation_loss, model_img, title, file_name)

    # Plot target distribution
    title = 'Distribution Target Validation Set %s' % model
    file_name = 'distribution-target-validation-%s' %model
    utils.plot_target_distribution(y_g, y_p, model_img, title, file_name)

    if not net_input == 'all_sensors':
        # Plot sensing histogram
        title = 'Sensing Validation Set %s' % model
        file_name = 'histogram-sensing-validation-%s' % model

        if communication:
            x_train = x_train.reshape(-1, 7)
            x = [x_train[0], x_train[1], x_train[2], x_train[3], x_train[4], x_train[5], x_train[6]]
        else:
            x = [x_train[0], x_train[1], x_train[2], x_train[3], x_train[4], x_train[5], x_train[6]]

        label = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']
        utils.my_histogram(x, 'sensing (%s)' % net_input, model_img, title, file_name, label)

    # Evaluate prediction of the learned controller to the omniscient groundtruth
    # Plot R^2 of the regressor between prediction and ground truth on the validation set
    title = 'Regression %s vs %s' % (model, dataset)
    file_name = 'regression-%s-vs-%s' % (model, dataset)

    x_label = 'groundtruth'
    y_label = 'prediction'
    if communication:
        y_g = np.reshape(np.array(y_g).flat, [-1])
        y_p = np.reshape(np.array(y_p).flat, [-1])

    utils.plot_regressor(y_g, y_p, x_label, y_label, model_img, title, file_name)


def controller_plots(model_dir, ds, ds_eval, groundtruth, prediction, communication):
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

    if not communication:
        groundtruth = np.array(groundtruth).flatten()
        prediction = np.array(prediction).flatten()

    x_label = 'groundtruth'
    y_label = 'prediction'
    utils.plot_regressor(groundtruth, prediction, x_label, y_label, model_img, title, file_name)


def evaluate_controller(model_dir, ds, ds_eval, groundtruth, sensing, net_input, communication):
    """

    :param model_dir:
    :param ds:
    :param ds_eval:
    :param groundtruth:
    :param sensing: used to obtain the prediction
    :param net_input
    :param communication
    """
    if communication:
        groundtruth = np.reshape(np.array(groundtruth).flat, [-1])
        sensing = np.reshape(np.array(sensing).flat, [-1, 7])

    controller_predictions = []
    controller = controllers.ManualController(net_input=net_input)

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

    controller_plots(model_dir, ds, ds_eval, groundtruth, controller_predictions, communication)


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
    controller = controllers.LearnedController(net=net, net_input=net_input)

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

        control, _ = controller.perform_control(state, dt=0.1)

        controller_predictions.append(control)

    title = 'Response %s - %s' % (model, net_title)
    file_name = 'response-%s-%s' % (model, net_title)

    # Plot the output of the network
    utils.plot_response(sensing, controller_predictions, x_label, model_img, title, file_name, index)


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
        return controllers.LearnedController(net=net, net_input=net_input, **kwargs)

    # FIXME do not use simulation
    world, myts = g.setup(controller_factory, myt_quantity)

    simulations = 17 * 10

    range = 2 * avg_gap

    x = np.linspace(0, range, num=simulations)
    control_predictions = []

    for simulation in tqdm.tqdm(x):
        g.init_positions(myts, net_input, avg_gap, variate_pose=True, x=simulation)

        world.step(dt=0.1)
        # myts[1].learned_controller()
        control = myts[1].motor_left_target
        control_predictions.append(control)

    title = 'Response %s by varying init position' % model
    file_name = 'response-%s-varying_init_position' % model

    # Plot the output of the network
    utils.plot_response(x, control_predictions, 'init avg gap', model_img, title, file_name)


def network_evaluation(indices, file_losses, runs_dir, model_dir, model, model_img, ds, ds_eval, communication, net_input, avg_gap):
    """

    :param indices:
    :param file_losses:
    :param runs_dir:
    :param model_dir:
    :param model:
    :param model_img:
    :param ds:
    :param ds_eval:
    :param communication:
    :param net_input:
    :param avg_gap:
    :return:
    """
    print('\nGenerating plots for %s…' % model)
    net = torch.load('%s/%s' % (model_dir, model), map_location='cpu')
    # Ensure that the network is loaded in evaluation mode by default.
    net.eval()

    train_indices, validation_indices, test_indices = indices[1]

    x_train, x_valid, x_test, \
    y_train, y_valid, y_test = utils.from_indices_to_dataset(runs_dir, train_indices, validation_indices,
                                                             test_indices, net_input, communication)

    # Load the metrics
    losses = pd.read_pickle(file_losses)
    training_loss = losses.loc[:, 't. loss']
    validation_loss = losses.loc[:, 'v. loss']

    prediction = net(torch.FloatTensor(x_valid))

    network_plots(model_img, ds, model, net_input, prediction, training_loss, validation_loss, x_train, y_valid, communication)

    # Evaluate prediction of the distributed controller with the omniscient groundtruth
    evaluate_controller(model_dir, ds, ds_eval, y_valid, x_valid, net_input, communication)

    if not communication:
        if not net_input == 'all_sensors':
            # Evaluate the learned controller by passing a specific input sensing configuration
            x, s = generate_sensing()
            sensing = np.stack([s, s, np.divide(x, 1000), s, s, s, s], axis=1)
            index = 2
    
            evaluate_net(model_img, model, net, net_input, 'net([0, 0, x, 0, 0, 0, 0])', sensing, index,
                         'center proximity sensor')

            index = -1
            sensing = np.stack([s, s, s, s, s, np.divide(x, 1000), np.divide(x, 1000)], axis=1)
            evaluate_net(model_img, model, net, net_input, 'net([0, 0, 0, 0, 0, x, x])', sensing, index,
                         'rear proximity sensors')

        # Evaluate the learned controller by passing a specific initial position configuration
        test_controller_given_init_positions(model_img, net, model, net_input, avg_gap)

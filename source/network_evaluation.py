import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils import data

from utils import my_plots
from utils import utils
from utils.utils import ThymioState


def binary_acc(y_pred, y_test):
    """
    :param y_pred: prediction
    :param y_test: target
    :return acc: accuracy
    """
    correct_results_sum = float(np.sum(y_pred == y_test))
    acc = correct_results_sum / y_test.shape[0]
    acc = np.round(acc * 100, 2)

    return acc


def network_plots(model_img, dataset, model, net_input, prediction, training_loss, validation_loss, x_train, y_valid,
                  communication, goal):
    """
    :param model_img: directory for the output image of the model
    :param dataset: name of the dataset
    :param model: netowork
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    :param prediction: output control of the model
    :param training_loss: error on the train set
    :param validation_loss: error on the validation set
    :param x_train: train input
    :param y_valid: validation output
    :param communication: states if the communication is used by the network
    :param goal: task to be performed (can be colour or distribute)
    """
    y_p = prediction.squeeze().tolist()
    y_g = y_valid.squeeze().tolist()

    # Plot train and validation losses
    title = 'Loss %s' % model
    file_name = 'loss-%s' % model
    my_plots.plot_losses(training_loss, validation_loss, model_img, title, file_name, goal)

    # Plot target distribution
    title = 'Distribution Target Validation Set %s' % model
    file_name = 'distribution-target-validation-%s' % model
    my_plots.plot_target_distribution(y_g, y_p, model_img, title, file_name)

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
        my_plots.my_histogram(x, 'sensing (%s)' % net_input, model_img, title, file_name, label)

    if not goal == 'colour':
        # Evaluate prediction of the learned controller to the omniscient groundtruth
        # Plot R^2 of the regressor between prediction and ground truth on the validation set
        title = 'Regression %s vs %s' % (model, dataset)
        file_name = 'regression-%s-vs-%s' % (model, dataset)

        x_label = 'groundtruth'
        y_label = 'prediction'
        if communication:
            y_g = np.reshape(np.array(y_g).flat, [-1])
            y_p = np.reshape(np.array(y_p).flat, [-1])

            y_g = y_g[~np.isnan(y_g)]
            y_p = y_p[~np.isnan(y_p)]

        my_plots.plot_regressor(y_g, y_p, x_label, y_label, model_img, title, file_name)


def controller_plots(model_dir, ds, ds_eval, groundtruth, prediction, communication):
    """

    :param model_dir: directory containing the trained model
    :param ds: name of the dataset
    :param ds_eval: name of the dataset  for the evaluation (usually the manual one)
    :param groundtruth: evidence
    :param prediction: output control
    :param communication: states if the communication is used by the network
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
    my_plots.plot_regressor(groundtruth, prediction, x_label, y_label, model_img, title, file_name)


def evaluate_controller(model_dir, ds, ds_eval, groundtruth, sensing, net_input, communication, goal, controllers):
    """

    :param model_dir: directory containing the trained model
    :param ds: name of the dataset
    :param ds_eval: name of the dataset  for the evaluation (usually the manual one)
    :param groundtruth: evidence
    :param sensing: used to obtain the prediction
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    :param communication: states if the communication is used by the network
    :param goal: task to perform (in this case distribute)
    :param controllers: reference to the controller class
    """
    if communication:
        groundtruth = np.reshape(np.array(groundtruth).flat, [-1])
        groundtruth = groundtruth[~np.isnan(groundtruth)]

        sensing = np.reshape(np.array(sensing).flat, [-1, sensing.shape[3]])

    controller_predictions = []
    controller = controllers.ManualController(net_input=net_input, name='manual', goal=goal, N=1)

    for sample in sensing:
        if np.isnan(sample).all():
            continue
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
                      'prox_comm': prox_comm, 'index': 1}

        state = ThymioState(state_dict)

        control, _ = controller.perform_control(state, dt=0.1)

        controller_predictions.append(control)

    controller_plots(model_dir, ds, ds_eval, groundtruth, controller_predictions, communication)


def generate_sensing():
    """

    :return x, s: x is the configuration, s is the zeros array
    """
    x = np.arange(4500)
    s = np.zeros(x.shape[0])

    return x, s


def evaluate_net(model_img, model, net, net_input, net_title, sensing, index, x_label, goal, communication, controllers):
    """

    :param model_img: directory for the output image of the model
    :param model: name of the model
    :param net: model used
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    :param net_title: network title
    :param sensing: input sensing
    :param index: 1D vector
    :param x_label: label of the x axis
    :param goal: task to perform (in this case distribute)
    :param communication: states if the communication is used by the network
    :param controllers: reference to the controller class
    """
    controller_predictions = []
    controller = controllers.LearnedController(net=net, net_input=net_input, name='learned', goal=goal, N=3,
                                               communication=communication)

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
    my_plots.plot_response(sensing, controller_predictions, x_label, model_img, title, file_name, index)


def network_evaluation(indices, file_losses, runs_dir, model_dir, model, model_img, ds, ds_eval, communication,
                       net_input, task='task1', runs_dir_manual=None, runs_dir_learned=None):
    """

    :param indices: sample indices
    :param file_losses: file where to save the metrics
    :param runs_dir: directory containing the simulation runs generated with the omniscient controller
    :param model_dir: directory containing the network data
    :param model: network
    :param model_img: directory for the output image of the model
    :param ds: name of the dataset
    :param ds_eval: name of the dataset  for the evaluation (usually the manual one)
    :param communication: states if the communication is used by the network
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    :param task: task to be performed
    :param runs_dir_manual: directory containing the simulation runs generated with the manual controller
    :param runs_dir_learned: directory containing the simulation runs generated with the learned controller
    """
    if task == 'task1':
        from controllers import controllers_task1 as controllers
        goal = 'distribute'
    else:
        from controllers import controllers_task2 as controllers
        goal = 'colour'

    print('\nGenerating plots for %s…' % model)
    net = torch.load('%s/%s' % (model_dir, model), map_location='cpu')
    # Ensure that the network is loaded in evaluation mode by default.
    net.eval()
    device = torch.device('cpu')

    train_indices, validation_indices, test_indices = indices[1]

    x_train, x_valid, x_test, \
    y_train, y_valid, y_test,\
    q_train, q_valid, q_test = utils.from_indices_to_dataset(runs_dir, train_indices, validation_indices,
                                                             test_indices, net_input, communication, goal)

    # Load the metrics
    losses = pd.read_pickle(file_losses)
    training_loss = losses.loc[:, 't. loss']
    validation_loss = losses.loc[:, 'v. loss']

    prediction = net(torch.FloatTensor(x_valid), torch.FloatTensor(q_valid))

    network_plots(model_img, ds, model, net_input, prediction, training_loss, validation_loss, x_train, y_valid, communication, goal)

    if goal == 'colour':
        x_label = 'groundtruth'
        y_label = 'prediction'

        # Evaluate prediction of the manual controller to the omniscient groundtruth
        # Plot R^2 of the regressor between prediction and ground truth on the validation set
        # Plot accuracy of manual controller
        y_target_m, y_m = utils.extract_targets(runs_dir_manual, validation_indices)

        acc = binary_acc(y_target_m, y_m)
        title = 'Accuracy %s vs %s' % (ds_eval, ds)
        file_name = 'accuracy-%s-vs-%s' % (ds_eval, ds)
        # TODO
        my_plots.plot_accuracy(y_target_m, y_m, acc, x_label, y_label, model_img, title, file_name)

        # Evaluate prediction of the learned controller to the omniscient groundtruth
        # Plot R^2 of the regressor between prediction and ground truth on the validation set
        # Plot accuracy of learned controller
        y_target_n, y_n = utils.extract_targets(runs_dir_learned, validation_indices)

        acc = binary_acc(y_target_n, y_n)
        title = 'Accuracy %s vs %s' % (model, ds)
        file_name = 'accuracy-%s-vs-%s' % (model, ds)
        # TODO
        my_plots.plot_accuracy(y_target_n, y_n, acc, x_label, y_label, model_img, title, file_name)

        # Model heatmap
        controller = controllers.LearnedController(net_input=net_input, name='learned', goal=goal, N=3,
                                                   net=net, communication=communication)

        input1 = np.linspace(0, 1)
        input2 = np.linspace(0, 1)
        inputs = np.array(np.meshgrid(input1, input2)).T.reshape(-1, 2)

        output1 = []
        output2 = []
        for i in range(len(inputs)):
            state_dict = {'sim': True, 'prox_values': np.zeros(7), 'prox_comm': np.zeros(7),
                          'messages': inputs[i], 'index': 1}

            state = ThymioState(state_dict)

            colour, comm = controller.perform_control(state, dt=0.1)

            output1.append(colour.item())
            output2.append(comm)

        heat1 = pd.DataFrame(np.hstack((inputs, np.atleast_2d(output1).T)), columns=['rear communication', 'front communication', 'colour'])
        heat2 = pd.DataFrame(np.hstack((inputs, np.atleast_2d(output2).T)), columns=['rear communication', 'front communication', 'transmitted communication'])
        # TODO
        my_plots.plot_heatmap(heat1, heat2, model_img)

        # Plot ROC curve and accuracy
        _, _, valid = utils.from_dataset_to_tensors(x_train, y_train, x_valid, y_valid, x_test, y_test, q_train,
                                                           q_valid, q_test)

        valid_minibatch = data.DataLoader(valid, batch_size=100, shuffle=False)
        probs = []
        targets = []

        with torch.no_grad():
            net.eval()

            for batch in valid_minibatch:
                inputs, labels, size = (tensor.to(device) for tensor in batch)
                outputs = net(inputs, size)

                output_flatten = torch.flatten(outputs)[~torch.isnan(torch.flatten(outputs))]
                labels_flatten = torch.flatten(labels)[~torch.isnan(torch.flatten(labels))]

                probs.append(output_flatten.tolist())
                targets.append(labels_flatten.tolist())

        probs = np.array(list(itertools.chain.from_iterable(probs)))
        targets = np.array(list(itertools.chain.from_iterable(targets)))

        fpr, tpr, thresholds = roc_curve(targets, probs)
        auc = roc_auc_score(targets, probs)
        filename = 'roc-%s' % model

        my_plots.plot_roc_curve(fpr, tpr, auc, acc, model_img, filename, model)

    else:
        # Evaluate prediction of the distributed controller with the omniscient groundtruth
        evaluate_controller(model_dir, ds, ds_eval, y_valid, x_valid, net_input, communication, goal, controllers)

        if not communication:
            if not net_input == 'all_sensors':
                # Evaluate the learned controller by passing a specific input sensing configuration
                x, s = generate_sensing()
                sensing = np.stack([s, s, np.divide(x, 1000), s, s, s, s], axis=1)
                index = 2

                evaluate_net(model_img, model, net, net_input, 'net([0, 0, x, 0, 0, 0, 0])', sensing, index,
                             'center proximity sensor', goal, communication, controllers)

                index = -1
                sensing = np.stack([s, s, s, s, s, np.divide(x, 1000), np.divide(x, 1000)], axis=1)
                evaluate_net(model_img, model, net, net_input, 'net([0, 0, 0, 0, 0, x, x])', sensing, index,
                             'rear proximity sensors', goal, communication, controllers)

from typing import AnyStr

import torch
import tqdm
from torch.utils import data

from networks.communication_network import Sync, CommunicationNetNoSensing
from networks.metrics import StreamingMean, NetMetrics
from utils import utils
from utils.utils import export_network


def train_net(epochs: int,
              train_dataset: data.TensorDataset,
              valid_dataset: data.TensorDataset,
              test_dataset: data.TensorDataset,
              net: torch.nn.Module,
              metrics_path: AnyStr,
              device,
              batch_size: int = 100,
              learning_rate: float = 0.01,
              criterion=torch.nn.BCELoss()
              ) -> NetMetrics:
    """
    :param epochs: number of epochs
    :param train_dataset: training dataset
    :param valid_dataset: validation dataset
    :param test_dataset: testing dataset
    :param net: model
    :param metrics_path: file where to save the metrics
    :param device: device used (cpu or cuda)
    :param batch_size: size of the batch (default: 100)
    :param learning_rate: learning rate (default: 0.01)
    :param criterion: loss function (default: ​Mean Squared Error)
    :return training_loss, validation_loss, testing_loss: output losses of the datasets
    """

    train_minibatch = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_minibatch = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_minibatch = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    # Support objects for metrics and validation
    training_loss = StreamingMean()

    t = tqdm.trange(epochs, unit='epoch')
    metrics = NetMetrics(t, metrics_path)

    for _ in tqdm.tqdm(range(epochs)):
        # Re-enable training mode, which is disabled by the evaluation
        # Turns on dropout, batch normalization updates, …
        net.train()
        training_loss.reset()

        for batch in train_minibatch:
            inputs, labels, size = (tensor.to(device) for tensor in batch)
            outputs = net(inputs, size)

            output_flatten = torch.flatten(outputs)[~torch.isnan(torch.flatten(outputs))]
            labels_flatten = torch.flatten(labels)[~torch.isnan(torch.flatten(labels))]
            loss = criterion(output_flatten, labels_flatten)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Accumulate metrics across batches
            training_loss.update(loss, inputs.shape[0])

        validation_loss = validate_net(net, device, valid_minibatch, criterion)

        # Record the metrics for the current epoch
        metrics.update(training_loss.mean, validation_loss)

    return metrics


def validate_net(net, device, valid_minibatch, criterion=torch.nn.MSELoss()):
    """
    :param net: model
    :param device: device used (cpu or cuda)
    :param valid_minibatch: validation batch
    :param criterion: loss function
    :return validation_loss.mean: resulting loss of the validation set
    """

    # Support objects for metrics and validation
    validation_loss = StreamingMean()

    with torch.no_grad():
        net.eval()
        validation_loss.reset()

        for batch in valid_minibatch:
            inputs, labels, size = (tensor.to(device) for tensor in batch)
            outputs = net(inputs, size)

            output_flatten = torch.flatten(outputs)[~torch.isnan(torch.flatten(outputs))]
            labels_flatten = torch.flatten(labels)[~torch.isnan(torch.flatten(labels))]
            loss = criterion(output_flatten, labels_flatten)

            validation_loss.update(loss, inputs.shape[0])

    return validation_loss.mean


def network_train(indices, file_losses, runs_dir, model_dir, model, communication, net_input, save_net):
    """
    Split the dataset also defining input and output, using the indices.
    Generate tensors.
    Create the neural network and optimizer, and set the device.
    Train the model and save it.

    :param indices: sample indices
    :param file_losses: file where to save the metrics
    :param runs_dir: directory containing the simulation runs
    :param model_dir: directory containing the network data
    :param model: network
    :param communication: states if the communication is used by the network
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    :param save_net: states if save or not the network
    """
    train_indices, validation_indices, test_indices = indices[1]

    # Split the dataset also defining input and output, using the indices
    x_train, x_valid, x_test, \
    y_train, y_valid, y_test, \
    q_train, q_valid, q_test = utils.from_indices_to_dataset(runs_dir, train_indices, validation_indices,
                                                             test_indices, net_input, communication, 'colour')

    # Generate the tensors
    test, train, valid = utils.from_dataset_to_tensors(x_train, y_train, x_valid, y_valid, x_test, y_test, q_train,
                                                       q_valid, q_test)

    print('\nTraining %s…' % model)
    # Create the neural network and optimizer
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    net = CommunicationNetNoSensing(x_train.shape[3], device=device, sync=Sync.sync)
    net.to(device)

    metrics = train_net(epochs=500,
                        train_dataset=train,
                        valid_dataset=valid,
                        test_dataset=test,
                        batch_size=10,
                        learning_rate=0.001,
                        net=net,
                        metrics_path=file_losses,
                        device=device)

    torch.save(net, '%s/%s' % (model_dir, model))
    metrics.finalize()

    if save_net:
        if net_input == 'all_sensors':
            width = 14
        else:
            width = 7

        if communication:
            dummy_input = torch.randn(1, 1, 1, width)
        else:
            dummy_input = torch.randn(1, width)

        export_network(model_dir, model, dummy_input, dummy_input)

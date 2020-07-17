from typing import AnyStr

import torch
import tqdm
from torch.utils import data

from networks.communication_network import Sync, CommunicationNet
from networks.distributed_network import DistributedNet
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
              criterion=torch.nn.MSELoss()
              ) -> NetMetrics:
    """
    :param epochs:
    :param train_dataset:
    :param valid_dataset:
    :param test_dataset:
    :param net:
    :param metrics_path:
    :param device
    :param batch_size:
    :param learning_rate:
    :param criterion:
    :return training_loss, validation_loss, testing_loss:

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
            inputs, labels = (tensor.to(device) for tensor in batch)
            output = net(inputs)

            loss = criterion(output, labels)

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
    :param net:
    :param device
    :param valid_minibatch:
    :param criterion:
    :return validation_loss.mean:
    """

    # Support objects for metrics and validation
    validation_loss = StreamingMean()

    with torch.no_grad():
        net.eval()
        validation_loss.reset()

        for batch in valid_minibatch:
            inputs, labels = (tensor.to(device) for tensor in batch)
            t_output = net(inputs)

            loss = criterion(t_output, labels)
            validation_loss.update(loss, inputs.shape[0])

    return validation_loss.mean


def network_train(indices, file_losses, runs_dir, model_dir, model, communication, net_input, save_net, task='task1'):
    """
    :param indices
    :param file_losses
    :param runs_dir:
    :param model_dir: directory containing the network data
    :param model
    :param communication
    :param net_input
    :param save_net
    :param task
    """
    train_indices, validation_indices, test_indices = indices[1]

    # Split the dataset also defining input and output, using the indices
    x_train, x_valid, x_test, \
    y_train, y_valid, y_test = utils.from_indices_to_dataset(runs_dir, train_indices, validation_indices,
                                                             test_indices, net_input, communication)

    # Generate the tensors
    test, train, valid = utils.from_dataset_to_tensors(x_train, y_train, x_valid, y_valid, x_test, y_test)

    print('\nTraining %s…' % model)
    # Create the neural network and optimizer
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')

    if communication:
        net = CommunicationNet(x_train.shape[3], device=device, sync=Sync.sync)
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
    else:
        net = DistributedNet(x_train.shape[1])
        net.to(device)

        metrics = train_net(epochs=50,
                            train_dataset=train,
                            valid_dataset=valid,
                            test_dataset=test,
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

        export_network(model_dir, model, dummy_input)

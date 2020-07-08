# communication_network.py
# © 2019-2020 Marco Verna
# © 2020 Giorgia Adorni
# Adapted from https://github.com/MarcoVernaUSI/distributed/blob/Refactored/com_network.py


from enum import Enum
from random import shuffle
from typing import Sequence, Tuple, TypeVar, Callable

import numpy as np
import torch
import torch.nn as nn

Sensing = TypeVar('Sensing')
Control = TypeVar('Control')
Communication = TypeVar('Communication')

ControlOutput = Tuple[Sequence[Control], Sequence[Communication]]
Controller = Callable[[Sequence[Sensing]], ControlOutput]


class Sync(Enum):
    random = 1
    sequential = 2
    sync = 3
    random_sequential = 4


class SingleNet(nn.Module, ):
    def __init__(self, input_size):
        """
        :param input_size: dimension of the sensing vector
        """
        super(SingleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size + 2, 22)  # (7 + 2) or (14 + 2)
        self.relu = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(22, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_):
        """
        :param input_: input of the network, vector containing the sensting and the messages received by the robot
                       (can be multidimensional, that means a row for each robot)
        :return output: output of the network containing the control and the message to communicate (shape: 1 x 2)
        """
        hidden = self.fc1(input_)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        # Convert the communication in values between 0 and 1 using the Sigmoid activation function
        speed = output[:, 0, None]
        communication = output[:, 1, None]

        communication = self.sigmoid(communication)

        output[:, 0, None] = speed
        output[:, 1, None] = communication

        return output


def init_comm(thymio: int, device):
    """
    Initialise the communication vector (for the initial timestep of each sequence)
    :param thymio: thymio quantity
    :param device
    :return: communication vector
    """
    out = np.zeros(thymio + 2)
    out[1:-1] = np.random.uniform(0, 1, thymio)

    return torch.Tensor(out, device=device)


def input_from(ss, comm, i, sim=False):
    """

    :param ss: sensing
    :param comm: communication vector
    :param i: index of the thymio in the row
    :param sim: boolean True if executed in simulation, False otherwise
    :return input_
    """
    if sim:
        input_ = torch.cat((ss, comm[i].view(1), comm[i+2].view(1)), 0)
    else:
        input_ = torch.cat((ss[i], comm[i].view(1), comm[i + 2].view(1)), 0)

    return input_


class CommunicationNet(nn.Module):
    def __init__(self, input_size, device, sync: Sync = Sync.sequential, module: nn.Module = SingleNet, input_fn=input_from) -> None:
        """

        :param input_size: dimension of the sensing vector (can be 7 or 14)
        :param device
        :param sync
        :param module
        :param input_fn
        """
        super(CommunicationNet, self).__init__()
        self.input_size = input_size
        self.single_net = module(self.input_size)
        self.device = device
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None

    def step(self, xs, comm, sync: Sync, sim=False, i=None):
        """

        :param xs: input sensing of a certain timestep (shape: 1 x N x sensing)
        :param comm: communication vector (shape: 1 X N)
        :param sync:
        :param sim: boolean true if step executed in simulation
        :param i: index of the thymio in the row
        :return control
        """
        if sync == Sync.sync:
            if sim:
                # if i == 0:
                #     comm[0] = 0
                # elif i == self.thymio - 1:
                #     comm[1] = 0

                input = self.input_fn(xs, comm, i, sim=True)
                input = input.unsqueeze(0)
            else:
                input = torch.stack([self.input_fn(xs, comm, i) for i in range(xs.shape[0])], 0)

            output = self.single_net(input)

            control = output[:, 0]

            if sim:
                comm[i + 1] = output[:, 1]
            else:
                comm[1:-1] = output[:, 1]
        else:
            # FIXME
            #  not working in simulation

            # random_sequential
            # shuffle for each sequence
            if sync == Sync.random_sequential:
                # keep the random indices list
                indices = self.tmp_indices
            # sequential
            else:
                # re-create the sorted indices list
                indices = list(range(xs.shape[0]))
            # random
            # shuffle for each timestep
            if sync == Sync.random:
                shuffle(indices)

            cs = []
            for i in indices:
                output = self.single_net(self.input_fn(xs, comm, i))
                comm[i + 1] = output[1]
                cs.append(output[:1])
            control = torch.cat(cs, 0)

        return control

    def forward(self, batch):
        """

        :param batch:
        :return: rd
        """
        rs = []
        for sequence in batch:
            # FIXME
            comm = init_comm(sequence[0].shape[0], device=self.device)
            controls = []
            tmp = list(range(sequence[0].shape[0]))
            shuffle(tmp)
            self.tmp_indices = tmp

            #  xs is the timestep
            for xs in sequence:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))

        return torch.stack(rs)

    def controller(self, thymio, sync: Sync = Sync.sync) -> Controller:
        """

        :param thymio: number of thymio in the simulation
        :param sync:
        :return f
        """
        if sync == None:
            sync = self.sync

        self.thymio = thymio

        tmp = list(range(self.thymio))
        shuffle(tmp)
        self.tmp_indices = tmp

        comm = init_comm(self.thymio, sim=True, device=self.device)

        def f(sensing: Sequence[Sensing], communication, i) -> Tuple[Sequence[Control], Sequence[float]]:
            """
            :param sensing:
            :param communication: array containing the communication received by the thymio from left and right
            :param i: index of the robot in the row
            :return control, new communication vector
            """
            nonlocal comm
            
            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)
                if communication is not None:
                    comm[i] = communication[0]
                    comm[i + 2] = communication[1]
                control = self.step(sensing, comm, sync=sync, sim=True, i=i).numpy()

                print("comm = ", comm)
                print("comm subset = ", comm[1:-1])
                print()

                return control, comm[1:-1].clone().numpy().flatten()

        return f

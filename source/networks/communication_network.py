# communication_network.py
# © 2019-2020 Marco Verna
# © 2020 Giorgia Adorni
# Adapted from https://github.com/MarcoVernaUSI/distributed/blob/Refactored/com_network.py


import random
from enum import Enum
from random import shuffle
from typing import Sequence, Tuple, TypeVar, Callable

import torch
import torch.nn as nn

Sensing = TypeVar('Sensing')
Control = TypeVar('Control')
Communication = TypeVar('Communication')

ControlOutput = Tuple[Sequence[Control], Sequence[Communication]]
Controller = Callable[[Sequence[Sensing]], ControlOutput]


class Sync(Enum):
    """

    """
    random = 1
    sequential = 2
    sync = 3
    random_sequential = 4


class SingleNet(nn.Module, ):
    def __init__(self, input_size):
        """

        """
        super(SingleNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size + 2, 22)  # (7 + 2) or (14 + 2)
        self.relu = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(22, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_):
        """

        :param input_:
        :return:
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


def init_comm(N: int, device):
    """
    Initialise the communication vector (for the initial timestep of each sequence)
    :param N: thymio quantity
    :param device
    :return: communication vector
    """
    ls = [0.0]
    for i in range(N):
        ls.append(random.uniform(0, 1))
    ls.append(0.0)

    return torch.Tensor(ls, device=device)


def input_from(ss, comm, i):
    """

    :param ss:
    :param comm:
    :param i:
    :return:
    """
    input_ = torch.cat((ss[i], comm[i].view(1), comm[i + 2].view(1)), 0)

    return input_


class CommunicationNet(nn.Module):
    def __init__(self, input_size, device, sync: Sync = Sync.sequential, module: nn.Module = SingleNet, input_fn=input_from) -> None:
        """

        :param input_size:
        :param sync:
        :param module:
        :param input_fn:
        """
        super(CommunicationNet, self).__init__()
        self.input_size = input_size
        self.single_net = module(self.input_size)
        self.device = device
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None

    def step(self, xs, comm, sync: Sync):
        """

        :param xs: 1 x N x sensing
        :param comm: 1 X N
        :param sync:
        :return control:
        """
        if sync == Sync.sync:
            input = torch.stack([self.input_fn(xs, comm, i) for i in range(xs.shape[0])], 0)
            output = self.single_net(input)

            control = output[:, 0]
            comm[1:-1] = output[:, 1]
        else:
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

    def forward(self, runs):
        """

        :param runs:
        :return:
        """
        rs = []
        # for each sequence in batch
        for run in runs:
            # FIXME
            comm = init_comm(run[0].shape[0], device=self.device)
            controls = []
            tmp = list(range(run[0].shape[0]))
            shuffle(tmp)
            self.tmp_indices = tmp

            #  for each timestep in sequence
            for xs in run:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))

        return torch.stack(rs)

    def controller(self, N=1, sync: Sync = Sync.sync) -> Controller:
        """
        
        :param N:
        :param sync:
        :return:
        """
        if sync == None:
            sync = self.sync

        # tmp = list(range(N))
        # shuffle(tmp)
        # self.tmp_indices = tmp
        self.tmp_indices = [N]
        comm = init_comm(N, device=self.device)
        print("initial comm = ", comm)

        def f(sensing: Sequence[Sensing]) -> Tuple[Sequence[Control], Sequence[float]]:
            """
            :param sensing:
            :return:
            """
            with torch.no_grad():
                sensing = [sensing]
                sensing = torch.FloatTensor(sensing)
                control = self.step(sensing, comm, sync=sync).numpy()
                return control, comm[1:-1].clone().numpy().flatten()

        return f

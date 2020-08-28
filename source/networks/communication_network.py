# communication_network.py
# © 2019-2020 Marco Verna
# © 2020 Giorgia Adorni
# Adapted from https://github.com/MarcoVernaUSI/distributed/blob/Refactored/com_network.py


from enum import Enum
from random import shuffle
from typing import Sequence, Tuple, TypeVar, Callable
import torch
import torch.nn as nn
from torch.distributions import uniform

Sensing = TypeVar('Sensing')
Control = TypeVar('Control')
Communication = TypeVar('Communication')

ControlOutput = Tuple[Sequence[Control], Sequence[Communication]]
Controller = Callable[[Sequence[Sensing]], ControlOutput]


class Sync(Enum):
    """
    Class used to define the synchronisation approach of the communication updates,
    that can be random, sequential, synchronous and random_sequential
    """
    random = 1
    sequential = 2
    sync = 3
    random_sequential = 4


def init_comm(thymio: int, distribution):
    """
    Initialise the communication vector (for the initial timestep of each sequence).

    :param thymio: number of agents
    :param distribution: distribution used for the initialisation
    :return out: communication vector
    """
    out = torch.zeros(thymio + 2)
    out[1:-1] = torch.flatten(distribution.sample(torch.Size([thymio])))

    return out


def input_from(ss, comm, i, sim=False):
    """
    Parse the sensing and the communication and prepare it to be the input on the single net.

    :param ss: sensing
    :param comm: communication vector
    :param i: index of the thymio in the row
    :param sim: boolean True if executed in simulation, False otherwise
    :return in_put: parsed input containing the sensing values and the communication received
    """
    if sim:
        in_put = torch.cat((ss, comm[i].view(1), comm[i + 2].view(1)), 0)
    else:
        in_put = torch.cat((ss[i], comm[i].view(1), comm[i + 2].view(1)), 0)

    return in_put


def input_from_no_sensing(_, comm, i, sim=None):
    """
    Prepare the communication vector to be the input on the single net.

    :param comm: communication vector
    :param i: index of the thymio in the row
    :return in_put: parsed input containing the communication received
    """
    in_put = torch.cat((comm[i].view(1), comm[i + 2].view(1)), 0)

    return in_put


class SingleNet(nn.Module):
    """
    Low-level module that works on the sensing and the communication received by a single agent (in a certain timestep),
    producing as output the control and the communication to transmit.

    :param input_size: dimension of the sensing vector
    """
    def __init__(self, input_size):
        super(SingleNet, self).__init__()
        # self.fc1 = torch.nn.Linear(input_size + 2, 22)  # (7 + 2) or (14 + 2)
        # self.tanh = torch.nn.Tanh()
        # self.fc2 = torch.nn.Linear(22, 2)
        # self.sigmoid = torch.nn.Sigmoid()

        self.fc1 = torch.nn.Linear(input_size + 2, 10)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, in_put):
        """
        :param in_put: input of the network, vector containing the sensing and the messages received by the robot
                       (can be multidimensional, that means a row for each robot)
        :return output: output of the network containing the control and the message to communicate (shape: 1 x 2)
        """
        # hidden = self.fc1(in_put)
        # tanh = self.tanh(hidden)
        # output = self.fc2(tanh)

        hidden = self.fc1(in_put)
        tanh = self.tanh(hidden)
        hidden2 = self.fc2(tanh)
        tanh2 = self.tanh(hidden2)
        output = self.fc3(tanh2)

        # Convert the communication in values between 0 and 1 using the Sigmoid activation function
        speed = output[:, 0, None]
        communication = output[:, 1, None]

        communication = self.sigmoid(communication)

        output[:, 0, None] = speed
        output[:, 1, None] = communication

        return output


class CommunicationNet(nn.Module):
    """
    High-level module that handle the sensing of the agents.

    :param input_size: dimension of the sensing vector (can be 7 or 14)
    :param device: device used (cpu or gpu)
    :param sync: kind of synchronisation
    :param module: SingleNet
    :param input_fn: input function

    :var self.tmp_indices: communication indices
    :var self.distribution: distribution used for the initialisation of the communication
    """
    def __init__(self, input_size, device, sync: Sync = Sync.sequential, module: nn.Module = SingleNet,
                 input_fn=input_from) -> None:
        super(CommunicationNet, self).__init__()
        self.input_size = input_size
        self.single_net = module(self.input_size)
        self.device = device
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None
        self.distribution = None

    def step(self, xs, comm, sync: Sync, sim=False, i=None):
        """
        .. danger::
            Sync of types 'random', 'sequential' and 'random_sequential'
            are actually not working in simulation.


        :param xs: input sensing of a certain timestep (shape: 1 x N x sensing)
        :param comm: communication vector (shape: 1 X N)
        :param sync: kind of synchronisation
        :param sim: boolean true if step executed in simulation
        :param i: index of the thymio in the row
        :return control: speed of the agent
        """
        if sync == Sync.sync:
            if sim:
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

    def forward(self, batch, batch_size):
        """

        :param batch: batch elements
        :param batch_size: batch mask
        :return robots_control: network output
        """
        robots_control = []
        self.distribution = uniform.Uniform(0, 1)
        for idx, sequence in enumerate(batch):
            max_shape_size = int(sequence[0].shape[0])
            true_shape_size = int(batch_size[idx][0][0])
            sequence = sequence[:, :true_shape_size, :]

            comm = init_comm(int(sequence[0].shape[0]), self.distribution)
            controls = []
            tmp = list(range(int(sequence[0].shape[0])))
            shuffle(tmp)
            self.tmp_indices = tmp

            #  xs is the timestep
            for xs in sequence:
                control = self.step(xs, comm, self.sync)
                # reapply padding
                control = torch.nn.functional.pad(control, pad=(0, max_shape_size - true_shape_size), mode='constant', value=float('nan'))
                controls.append(control)

            robots_control.append(torch.stack(controls))

        robots_control = robots_control

        return torch.stack(robots_control)

    def controller(self, thymio, sync: Sync = Sync.sync) -> Controller:
        """

        :param thymio: number of agents in the simulation
        :param sync: kind of synchronisation
        :return f: controller function
        """
        if sync == None:
            sync = self.sync

        self.thymio = thymio

        tmp = list(range(self.thymio))
        shuffle(tmp)
        self.tmp_indices = tmp

        comm = init_comm(self.thymio, self.distribution)

        def f(sensing: Sequence[Sensing], communication, i) -> Tuple[Sequence[Control], Sequence[float]]:
            """
            :param sensing: input sensing
            :param communication: array containing the communication received by the thymio from left and right
            :param i: index of the robot in the row
            :return control, new communication vector: speed and complete communication vector
            """
            nonlocal comm

            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)
                if communication is not None:
                    comm[i] = communication[0]
                    comm[i + 2] = communication[1]
                control = self.step(sensing, comm, sync=sync, sim=True, i=i).numpy()

                return control, comm[1:-1].clone().numpy().flatten()

        return f


class SingleNetNoSensing(nn.Module):
    """
    Low-level module that works on the communication received by a single agent (in a certain timestep),
    producing as output the communication to transmit and the probability of a certain colour.

    :param _: dimension of the sensing vector

    """
    def __init__(self, _):

        super(SingleNetNoSensing, self).__init__()

        self.fc1 = torch.nn.Linear(2, 10)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, in_put):
        """

        :param in_put: input of the network, vector containing the messages received by the robot
                       (can be multidimensional, that means a row for each robot)
        :return output: output of the network containing the message to communicate
                        and the probability of a certain colour (shape: 1 x 2)
        """
        hidden = self.fc1(in_put)
        tanh = self.tanh(hidden)
        hidden2 = self.fc2(tanh)
        tanh2 = self.tanh(hidden2)
        output = self.fc3(tanh2)

        # Convert the communication and the colour in values between 0 and 1 using the Sigmoid activation function
        output = self.sigmoid(output)

        return output


class CommunicationNetNoSensing(CommunicationNet):
    """
    High-level module that handle the sensing of the agents.

    :param input_size: dimension of the sensing vector (can be 7 or 14)
    :param device: device used (cpu or gpu)
    :param sync: kind of synchronisation
    :param module: SingleNetNoSensing
    :param input_fn: input function

    :var self.tmp_indices: communication indices
    :var self.distribution: distribution used for the initialisation of the communication
    """
    def __init__(self, input_size, device, sync: Sync = Sync.sequential, module: nn.Module = SingleNetNoSensing,
                 input_fn=input_from_no_sensing) -> None:
        super(CommunicationNet, self).__init__()
        self.input_size = input_size
        self.single_net = module(self.input_size)
        self.device = device
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None
        self.distribution = None

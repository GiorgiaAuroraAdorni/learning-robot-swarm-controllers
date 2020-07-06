import random
from enum import Enum
from random import shuffle
from typing import Sequence, Tuple, TypeVar, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

State = TypeVar('State')
Sensing = TypeVar('Sensing')
Control = TypeVar('Control')
Communication = TypeVar('Communication')
MState = Sequence[State]
MSensing = Sequence[Sensing]
MControl = Sequence[Control]

Dynamic = Callable[[Sequence[State], Sequence[Control]], MState]
Sensor = Callable[[MState], MSensing]
ControlOutput = Tuple[Sequence[Control], Sequence[Communication]]
Controller = Callable[[Sequence[State], Sequence[Sensing]], ControlOutput]


class Sync(Enum):
    """

    """
    random = 1
    sequential = 2
    sync = 3
    random_sequential = 4


class SNet(nn.Module):
    def __init__(self):
        """

        """
        super(SNet, self).__init__()
        self.l1 = nn.Linear(4, 10)  # nn.Linear(7+2, 10)
        self.l2 = nn.Linear(10, 2)

    def forward(self, input_):
        """

        :param input_:
        :return:
        """
        ys = F.torch.tanh(self.l1(input_))
        return self.l2(ys)


# def input_from(ss, comm, i):
#     return torch.cat((ss[i], comm[i:i + 1], comm[i + 2:i + 3]), 0)


def init_comm(N: int):
    """
    Initialise the communication vector (for the initial timestep of each sequence)
    :param N: thymio quantity
    :return: communication vector
    """
    ls = [0.0]
    for i in range(N):
        ls.append(random.uniform(0, 1))
    ls.append(0.0)

    return Variable(torch.Tensor(ls))

def input_from(ss, comm, i):
    """

    :param ss:
    :param comm:
    :param i:
    :return:
    """
    return torch.cat((ss[i], comm[i], comm[i + 2]), 0)
    # FIXME
    # return torch.cat((ss[i], comm[i:i+1], comm[i+2:i+3]), 0)


class ComNet(nn.Module):
    def __init__(self, N: int, sync: Sync = Sync.sequential, module: nn.Module = SNet, input_fn=input_from) -> None:
        """

        :param N:
        :param sync:
        :param module:
        :param input_fn:
        """
        super(ComNet, self).__init__()
        self.single_net = module()
        self.N = N
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
            input = torch.stack([self.input_fn(xs, comm, i) for i in range(self.N)], 0)
            output = self.single_net(input)
            control = output[:, 0]
            # TODO vedi la matricina
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
                indices = list(range(self.N))
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
            comm = init_comm(self.N)
            controls = []
            tmp = list(range(self.N))
            shuffle(tmp)
            self.tmp_indices = tmp

            #  for each timestep in sequence
            for xs in run:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))
        return torch.stack(rs)

    def controller(self, sync: Sync = Sync.sync) -> Controller:
        """

        :param sync:
        :return:
        """
        N = self.N
        if sync == None:
            sync = self.sync
        tmp = list(range(self.N))
        shuffle(tmp)
        self.tmp_indices = tmp
        comm = init_comm(N)
        print("initial comm = ", comm)

        def f(sensing: Sequence[Sensing]) -> Tuple[Sequence[Control], Sequence[float]]:
            """
            :param sensing:
            :return:
            """
            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)
                control = self.step(sensing, comm, sync=sync).numpy()
                return control, comm[1:-1].clone().numpy().flatten()

        return f

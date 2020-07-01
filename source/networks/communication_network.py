from enum import Enum
from random import shuffle
from typing import Sequence, Tuple, TypeVar, Callable
import numpy as np

import random
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
    random = 1
    sequential = 2
    sync = 3
    random_sequential = 4


class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 2)

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        return self.l2(ys)


class SNetL(nn.Module):
    def __init__(self):
        super(SNetL, self).__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 2)
        self.out = torch.nn.Sigmoid()

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        ys2 = self.out(self.l2(ys))
        return ys2


class SNetLnoSensing(nn.Module):
    def __init__(self):
        super(SNetLnoSensing, self).__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 2)
        self.out = torch.nn.Sigmoid()

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        ys2 = self.out(self.l2(ys))
        return ys2


class S2Net(nn.Module):
    def __init__(self):
        super(S2Net, self).__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 3)

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        return self.l2(ys)


class SNetLeft(nn.Module):
    def __init__(self):
        super(SNetLeft, self).__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 2)

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        return self.l2(ys)


def input_from(ss, comm, i):
    return torch.cat((ss[i], comm[i:i + 1], comm[i + 2:i + 3]), 0)


def input_from_no_sensing(ss, comm, i):
    return torch.cat((comm[i:i + 1], comm[i + 2:i + 3]), 0)


def input_from2(ss, comm, i):
    return torch.cat((ss[i], comm[(2 * i):(2 * i) + 1], comm[(2 * i) + 3:(2 * i) + 4]), 0)


def input_from_left(ss, comm, i):
    return torch.cat((ss[i][:1], comm[i:i + 1]), 0)


def init_comm_zero(N: int):
    return Variable(torch.Tensor([0] * (N + 2)))


def init_comm(N: int):
    ls = [0.0]
    for i in range(N):
        ls.append(random.uniform(0, 1))
    ls.append(0.0)
    return Variable(torch.Tensor(ls))


class ComNet(nn.Module):
    def __init__(self, N: int, sync: Sync = Sync.sequential, module: nn.Module = SNet,
                 input_fn=input_from) -> None:
        super(ComNet, self).__init__()
        self.single_net = module()
        self.N = N
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None

    def step(self, xs, comm, sync: Sync):
        if sync == Sync.sync:
            input = torch.stack([self.input_fn(xs, comm, i) for i in range(self.N)], 0)
            output = self.single_net(input)
            control = output[:, 0]
            comm[1:-1] = output[:, 1]
        else:
            if sync == Sync.random_sequential:
                indices = self.tmp_indices
            else:
                indices = list(range(self.N))
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
        rs = []
        for run in runs:
            comm = init_comm(self.N)
            controls = []
            tmp = list(range(self.N))
            shuffle(tmp)
            self.tmp_indices = tmp
            for xs in run:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))
        return torch.stack(rs)

    def controller(self, sync: Sync = Sync.sync) -> Controller:
        N = self.N
        if sync == None:
            sync = self.sync
        tmp = list(range(self.N))
        shuffle(tmp)
        self.tmp_indices = tmp
        comm = init_comm(N)
        print("initial comm = ", comm)

        def f(state: Sequence[State], sensing: Sequence[Sensing]
              ) -> Tuple[Sequence[Control], Sequence[float]]:
            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)
                control = self.step(sensing, comm, sync=sync).numpy()
                return control, comm[1:-1].clone().numpy().flatten()

        return f


class Com2Net(nn.Module):
    def __init__(self, N: int, sync: Sync = Sync.sequential, module: nn.Module = S2Net,
                 input_fn=input_from2) -> None:
        super(Com2Net, self).__init__()
        self.single_net = module()
        self.N = N
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None

    def step(self, xs, comm, sync: Sync):
        # il sync va aggiornato ancora con la doppia comunicazione
        if sync == Sync.sync:
            input = torch.stack([self.input_fn(xs, comm, i) for i in range(self.N)], 0)
            output = self.single_net(input)
            control = output[:, 0]
            comm[1:-1] = output[:, 1]
        else:
            if sync == Sync.random_sequential:
                indices = tmp_indices
            else:
                indices = list(range(self.N))
            if sync == Sync.random:
                shuffle(indices)
            cs = []
            for i in indices:
                output = self.single_net(self.input_fn(xs, comm, i))
                comm[(2 * i) + 1:(2 * i) + 3] = output[1:]
                cs.append(output[:1])
            control = torch.cat(cs, 0)
        return control

    def forward(self, runs):
        rs = []
        for run in runs:
            comm = init_comm(self.N * 2)
            controls = []
            tmp = list(range(self.N))
            shuffle(tmp)
            self.tmp_indices = tmp
            for xs in run:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))
        return torch.stack(rs)

    def controller(self, sync: Sync = Sync.sequential) -> Controller:
        N = self.N
        if sync == None:
            sync = self.sync
        tmp = list(range(self.N))
        shuffle(tmp)
        self.tmp_indices = tmp
        comm = init_comm(N * 2)
        print("initial comm = ", comm)

        def f(state: Sequence[State], sensing: Sequence[Sensing]
              ) -> Tuple[Sequence[Control], Sequence[float]]:
            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)
                control = self.step(sensing, comm, sync=sync).numpy()
                return control, comm[1:-1].clone().numpy().flatten()

        return f


class ComNetL(ComNet):
    def __init__(self, N: int, sync: Sync = Sync.sequential, module: nn.Module = SNetL,
                 input_fn=input_from) -> None:
        super(ComNetL, self).__init__(N, sync, module, input_fn)
        self.single_net = module()
        self.N = N
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None


class ComNetLnoSensing(ComNet):
    def __init__(self, N: int, sync: Sync = Sync.sequential, module: nn.Module = SNetLnoSensing,
                 input_fn=input_from_no_sensing) -> None:
        super(ComNetLnoSensing, self).__init__(N, sync, module, input_fn)
        self.single_net = module()
        self.N = N
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None


################# forse non serve

class ComNetLnoSensingN(ComNetLnoSensing):
    def __init__(self, N, sync: Sync = Sync.sequential, module: nn.Module = SNetLnoSensing,
                 input_fn=input_from_no_sensing) -> None:
        super(ComNetLnoSensingN, self).__init__(N, sync, module, input_fn)
        self.single_net = module()
        self.Ns = N
        self.N = None
        self.sync = sync
        self.input_fn = input_fn
        self.tmp_indices = None

    def forward(self, runs):
        rs = []
        for run in runs:
            run_, actualN = self.unmask(run)
            self.N = actualN
            comm = init_comm(self.N)
            controls = []
            tmp = list(range(self.N))
            shuffle(tmp)
            self.tmp_indices = tmp
            for xs in run_:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))
        return rs

    def controller(self, sync: Sync = Sync.sync) -> Controller:
        N = self.N
        if sync == None:
            sync = self.sync
        tmp = list(range(self.N))
        shuffle(tmp)
        self.tmp_indices = tmp
        comm = init_comm(N)
        print("initial comm = ", comm)

        def f(state: Sequence[State], sensing: Sequence[Sensing]
              ) -> Tuple[Sequence[Control], Sequence[float]]:
            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)
                control = self.step(sensing, comm, sync=sync).numpy()
                return control, comm[1:-1].clone().numpy().flatten()

        return f

    def unmask(self, run):
        new = []
        for i in range(run.shape[0]):
            indices = np.where(run[i][:, 0] < 0)
            new.append(np.delete(run[i], indices, axis=0))
        new = torch.stack(new)
        actualN = new[0].shape[0]

        return new, actualN
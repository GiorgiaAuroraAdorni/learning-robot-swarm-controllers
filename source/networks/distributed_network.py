# distributed_network.py
# © 2019-2020 Marco Verna
# © 2020 Giorgia Adorni
# Adapted from https://github.com/MarcoVernaUSI/distributed/blob/Refactored/network.py


from typing import TypeVar, Callable, Tuple, Sequence

import torch

State = TypeVar('State')
Sensing = TypeVar('Sensing')
Control = TypeVar('Control')
Communication = TypeVar('Communication')

ControlOutput = Tuple[Sequence[Control], Sequence[Communication]]
Controller = Callable[[Sequence[Sensing]], ControlOutput]


class DistributedNet(torch.nn.Module):

    def __init__(self, input_size):
        super(DistributedNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 42)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(42, 1)

    def forward(self, xs):
        """

        :param xs:
        :return output
        """
        hidden = self.fc1(xs)
        relu = self.relu(hidden)
        output = self.fc2(relu)

        return output

    def controller(self, N=1) -> Controller:
        """

        :return f
        """

        def f(sensing: Sequence[Sensing]) -> Tuple[Sequence[Control]]:
            with torch.no_grad():
                return self(torch.FloatTensor(sensing)).numpy().flatten(),

        return f

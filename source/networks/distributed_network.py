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
    """
    Simplified “distributed network”, that takes as input an array containing the
    response values of the sensors – which can be either ``​prox_values​``, ``​prox_comm​``
    or ``all_sensors​`` – and the output is an array of 1 float that represents the ​speed​ of the wheels,
    which is assumed to be the same both right and left.

    The architecture of the network is straightforward:
    there are three linear layers each of size (inpuz_size, 10), (10, 10) and (10, 1) respectively,
    where input_size​ is the sum of the shape of the sensing – 7 or 14.
    The Tanh non-linear activation function is applied to the first and the second layer. ​

    :param input_size: dimension of the sensing vector
    """

    def __init__(self, input_size):
        super(DistributedNet, self).__init__()
        # self.fc1 = torch.nn.Linear(input_size, 22)
        # self.tanh = torch.nn.Tanh()
        # self.fc2 = torch.nn.Linear(22, 1)

        self.fc1 = torch.nn.Linear(input_size, 10)
        self.tanh = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 1)

    def forward(self, xs, _):
        """

        :param xs: input sensing batch
        :param _: mask of the batch
        :return output: speed
        """
        # hidden = self.fc1(xs)
        # tanh = self.tanh(hidden)
        # output = self.fc2(tanh)
        #
        # return torch.squeeze(output)
        hidden = self.fc1(xs)
        tanh = self.tanh(hidden)
        hidden2 = self.fc2(tanh)
        tanh2 = self.tanh(hidden2)
        output = self.fc3(tanh2)

        return torch.squeeze(output)

    def controller(self, thymio) -> Controller:
        """

        :param thymio: unused
        :return f: controller function
        """

        def f(sensing: Sequence[Sensing]) -> Tuple[Sequence[Control]]:
            with torch.no_grad():
                return self(torch.FloatTensor(sensing), 0).numpy().flatten(),

        return f

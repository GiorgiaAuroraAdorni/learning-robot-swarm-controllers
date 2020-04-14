import torch
from typing import Sequence, TypeVar, Callable, List, Tuple, Sequence, Optional
import torch.nn.functional as nn_funct

State = TypeVar('State')
Sensing = TypeVar('Sensing')
Control = TypeVar('Control')
Communication = TypeVar('Communication')

ControlOutput = Tuple[Sequence[Control], Sequence[Communication]]
Controller = Callable[[Sequence[State], Sequence[Sensing]], ControlOutput]


class DistributedNet(torch.nn.Module):

    def __init__(self, input_size):
        super(DistributedNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 10)
        self.l2 = torch.nn.Linear(10, 1)

    def forward(self, xs):
        ys = nn_funct.torch.tanh(self.l1(xs))
        return self.l2(ys)

    def controller(self) -> Controller:
        def f(state: Sequence[State], sensing: Sequence[Sensing]) -> Tuple[Sequence[Control]]:
            with torch.no_grad():
                return self(torch.FloatTensor(sensing)).numpy().flatten(),
        return f

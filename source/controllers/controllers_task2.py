import numpy as np

from utils import utils


class OmniscientController:
    """
    The robots can turn on and colour the top led also following an optimal “omniscient” controller.
    In this case, based on the position of the robots in the line, the omniscient control colour the robots in the first
    half with a colour and the others with another.
    """

    def __init__(self, name, goal, N, **kwargs):
        """

        :param name
        :param goal
        :param N: number of thymios in the simulation
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.name = name
        self.goal = goal
        self.N = N

    def perform_control(self, state, dt):
        """
        Do not move the robots but just colour the robots belonging to the first half with a certain colour and
        the other half with a different colour.
        :param state
        :param dt
        """

        return state.goal_colour

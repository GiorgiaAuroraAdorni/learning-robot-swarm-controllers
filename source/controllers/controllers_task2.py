import numpy as np

from utils import utils


class ManualController:
    """
    The robots are moved following a “distributed controller”, that is a simple proportional controller PID,
    with P fixed to -0.01, whose goal is to align the robots by minimizing the difference between the values
    ​​recorded by the front and rear sensors.
    """

    def __init__(self, name, goal, N, net_input, **kwargs):
        """

        :param name
        :param goal
        :param N: number of thymios in the simulation
        :param net_input:
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.name = name
        self.goal = goal
        self.N = N

        self.net_input = net_input

    def neighbors_distance(self, state):
        """
        Check if there is a robot ahead using the infrared sensor 2 (front-front).
        Check if there is a robot ahead using the infrared sensor 5 (back-left) and 6 (back-right).
        :param state
        :return back, front: response values of the rear and front sensors
        """
        sensing = utils.get_input_sensing(self.net_input, state, normalise=False)

        front = sensing[2]
        back = np.mean(np.array([sensing[5], sensing[6]]))

        return back, front

    def perform_control(self, state, dt):
        """
        Using the sensing of thw robots, decide which robots are the first and the last and start sending a
        communication message containing a value that is incremented by each robot that received it.
        Each robot can receive two messages, if the count received from the right is higher than the one from the left,
        then the agent is in the first half, otherwise it is in the second.
        When the robot is sure about is position (it is able to understand on which side of the row it is, compared to
        the mid of the line) then the robot can turn on the top led using different colours depending if it is
        positioned in the first or second half of the row.
        :param state:
        :param dt: control step duration

        """

        back, front = self.neighbors_distance(state)

        if state.goal_position[0] == state.position[0]:
            message = 1
            state.prox_comm_enable = True
            state.prox_comm_tx = message

            if back == 0:
                colour = 1

            elif front == 0:
                colour = 0

            else:
                raise ValueError('Invalid values for back and front!')

        else:
            communication = utils.get_received_communication(state, goal=self.goal)

            if communication == [0, 0]:
                # if no communication is received yet do not send any message and colour the top led randomly
                colour = np.random.randint(2)
                message = None
            elif communication[0] == 0:
                message = communication[1] + 1
                colour = 0
            elif communication[1] == 0:
                message = communication[0] + 1
                colour = 1
            else:
                if communication[0] > communication[1]:
                    colour = 0
                    message = communication[1] + 1
                elif communication[0] < communication[1]:
                    colour = 1
                    message = communication[0] + 1
                else:  # only if the number of robot is odd
                    colour = 1
                    message = communication[0] + 1

            if message is not None:
                state.prox_comm_enable = True
                state.prox_comm_tx = message

        return colour


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

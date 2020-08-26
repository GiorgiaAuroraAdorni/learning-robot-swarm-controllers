from utils import utils


class ManualController:
    """
    Using the sensing of the robots, decide which robots are the first and the last and start sending a
    communication message containing a value that is incremented by each robot that received it.
    Each robot can receive two messages, if the count received from the right is higher than the one from the left,
    then the agent is in the first half, otherwise it is in the second.
    When the robot is sure about is position (it is able to understand on which side of the row it is, compared to
    the mid of the line) then the robot can turn on the top led using different colours depending if it is
    positioned in the first or second half of the row.

    :param name: name of the controller used (in this case manual)
    :param goal: task to perform (in this case colour)
    :param N: number of thymios in the simulation
    :param kwargs: other arguments
    """

    def __init__(self, name, goal, N, net_input, **kwargs):
        super().__init__(**kwargs)

        self.name = name
        self.goal = goal
        self.N = N
        self.net_input = net_input

    def perform_control(self, state, dt):
        """
        :param state: object containing the agent information
        :param dt: control step duration

        :return colour, communication: the colour and the message to communicate
        """

        if state.index == 0:
            colour = 1
            message = 1

        elif state.index == self.N - 1:
            colour = 0
            message = 1

        else:
            communication = utils.get_received_communication(state, goal=self.goal)

            # if no communication is received yet, do not send any message and colour the top led randomly
            if communication == [0, 0]:
                colour = 2
                message = 0

            # if some communication is received...
            else:
                # if the number of robots is odd
                if self.N % 2 == 1:
                    # if no communication is received from left...
                    if communication[0] == 0:
                        if communication[1] > self.N // 2:
                            message = communication[1] - 1
                            colour = 1
                        elif communication[1] == self.N // 2:
                            message = communication[1] + 1
                            colour = 1
                        else:
                            message = communication[1] + 1
                            colour = 0

                    # if no communication is received from right...
                    elif communication[1] == 0:
                        if communication[0] > self.N // 2:
                            message = communication[0] - 1
                            colour = 0
                        elif communication[0] == self.N // 2:
                            message = communication[0] + 1
                            colour = 1
                        else:
                            message = communication[0] + 1
                            colour = 1

                    # if the communication is received from both sides...
                    else:
                        if communication[0] > communication[1]:
                            message = communication[1] + 1
                            colour = 0
                        else:
                            message = communication[0] + 1
                            colour = 1

                # if the number of robots is even
                elif self.N % 2 == 0:
                    # if no communication is received from left...
                    if communication[0] == 0:
                        if communication[1] > self.N // 2:
                            # WARNING
                            #  message = communication[1] or communication[1] - 1
                            message = communication[1]
                            colour = 1
                        else:
                            message = communication[1] + 1
                            colour = 0

                    # if no communication is received from right...
                    elif communication[1] == 0:
                        if communication[0] < self.N // 2:
                            message = communication[0] + 1
                            colour = 1
                        else:
                            # WARNING
                            #  message = communication[0] or communication[0] - 1
                            message = communication[0]
                            colour = 0

                    # if the communication is received from both sides...
                    else:
                        if communication[0] > communication[1]:
                            message = communication[1] + 1
                            colour = 0
                        elif communication[0] < communication[1]:
                            message = communication[0] + 1
                            colour = 1
                        else:
                            message = communication[0]
                            colour = 0

        return colour, int(message)


class OmniscientController:
    """
    The robots can turn on and colour the top led also following an optimal “omniscient” controller.
    In this case, based on the position of the robots in the line, the omniscient control colour the robots in the first
    half with a colour and the others with another.

    :param name: name of the controller used (in this case omniscient)
    :param goal: task to perform (in this case colour)
    :param N: number of agents in the simulation
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    :param kwargs: other arguments
    """

    def __init__(self, name, goal, N, net_input, **kwargs):
        super().__init__(**kwargs)

        self.name = name
        self.goal = goal
        self.N = N
        self.net_input = net_input

    def perform_control(self, state, dt):
        """
        Do not move the robots but just colour the robots belonging to the first half with a certain colour and
        the other half with a different colour.

        :param state: object containing the agent information
        :param dt: control step duration

        :return colour, communication: the colour and the message to communicate
        """

        return state.goal_colour, 0


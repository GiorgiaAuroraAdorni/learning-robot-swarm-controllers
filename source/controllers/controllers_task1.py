import os

import numpy as np

from controllers.pid import PID
from utils import utils


class ManualController:
    """
    The robots are moved following a “distributed controller”, that is a simple proportional controller PID,
    with P fixed to -0.01, whose goal is to align the robots by minimizing the difference between the values
    ​​recorded by the front and rear sensors.

    :param name: name of the controller used (in this case manual)
    :param goal: task to perform (in this case distribute)
    :param N: number of agents in the simulation
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    :param kwargs: other arguments

    :var p_distributed_controller: a simple proportional controller that returns the speed to apply
    """

    def __init__(self, name, goal, N, net_input, **kwargs):
        super().__init__(**kwargs)

        self.name = name
        self.goal = goal
        self.N = N

        self.p_distributed_controller = PID(5, 0, 0, max_out=16.6, min_out=-16.6)
        self.net_input = net_input

    def neighbors_distance(self, state):
        """
        Check if there is a robot ahead using the infrared sensor 2 (front-front)
        [2 and 9 in case of all sensors].
        Check if there is a robot ahead using the infrared sensor 5 (back-left) and 6 (back-right)
        [5, 6 and 12, 13 in case of all sensors].
        The distance from the neighbors is computed using a mapping function obtained experimentally that converts the
        intensities values contained in the sensor readings into distances.

        .. note:: The response values are actually intensities: the front correspond to the
                  frontal center sensor and the back to the mean of the response values of the rear sensors.


        :param state: object containing the agent information
        :return back, front: back distance and front distance of the thymio from the others
        """
        sensing = utils.get_input_sensing(self.net_input, state, normalise=False)

        if sensing == [0] * len(sensing):
            return 0, 0

        file = os.path.join('source/controllers', 'sensing_to_distances.pkl')
        distances, front_prox_values, back_prox_values, front_prox_comm, back_prox_comm = np.load(file, allow_pickle=True)

        f_pv_indices = np.nonzero(front_prox_values)[0]
        f_pv_indices = f_pv_indices[np.argsort(front_prox_values[f_pv_indices])]

        b_pv_indices = np.nonzero(back_prox_values)[0]
        b_pv_indices = b_pv_indices[np.argsort(back_prox_values[b_pv_indices])]

        f_pc_indices = np.nonzero(front_prox_comm)[0]
        f_pc_indices = f_pc_indices[np.argsort(front_prox_comm[f_pc_indices])]

        b_pc_indices = np.nonzero(back_prox_comm)[0]
        b_pc_indices = b_pc_indices[np.argsort(back_prox_comm[b_pc_indices])]

        front_distance_pv = np.interp(sensing[2], front_prox_values[f_pv_indices], distances[f_pv_indices])
        back_distance_pv = np.interp(np.mean(np.array([sensing[5], sensing[6]])), back_prox_values[b_pv_indices], distances[b_pv_indices])

        if sensing[7:] == [0] * 7 or self.net_input != 'all_sensors':
            return back_distance_pv, front_distance_pv

        elif sensing[0:7] == [0] * 7:
            front_distance_pc = np.interp(sensing[9], front_prox_comm[f_pc_indices], distances[f_pc_indices])
            back_distance_pc = np.interp(np.mean(np.array([sensing[12], sensing[13]])), back_prox_comm[b_pc_indices],
                                         distances[b_pc_indices])
            return back_distance_pc, front_distance_pc

        elif self.net_input == 'all_sensors':
            front_distance_pc = np.interp(sensing[9], front_prox_comm[f_pc_indices], distances[f_pc_indices])
            back_distance_pc = np.interp(np.mean(np.array([sensing[12], sensing[13]])), back_prox_comm[b_pc_indices], distances[b_pc_indices])

            if sensing[2] == 0.0:
                front = front_distance_pc
            else:
                front = front_distance_pv

            if np.mean([sensing[5], sensing[6]]) == 0.0:
                back = back_distance_pc
            elif np.mean([sensing[12], sensing[13]]) == 0.0:
                back = back_distance_pv
            else:
                back = np.mean([back_distance_pv, back_distance_pc])

            return back, front
        else:
           raise ValueError('Impossible values for sensing.')

    def compute_difference(self, state):
        """
        .. note:: Apply a small correction to the distance measured by the rear sensors since the front sensor used
                  is at a different x coordinate from the point to which the rear sensor of the robot that follows
                  points. This is because of the curved shape of the face of the Thymio.
                  The final difference is computed as follow:

                  .. math:: out = front - (back - correction)



        :param state: object containing the agent information
        :return out: the difference between the front and the rear distances
        """

        back, front = self.neighbors_distance(state)
        if back == 0 and front == 0:
            return 0

        # compute the correction
        delta_x = 7.41150769
        x = 7.95

        # Maximum possible response values
        # delta_x_m = 4505 * delta_x / 14
        # x_m = 4505 * x / 14
        # correction = x_m - delta_x_m
        correction = x - delta_x

        out = front - (back - correction)

        return out

    def perform_control(self, state, dt):
        """
        Keeping still the robots at the ends of the line, moves the others using the distributed controller,
        setting the ``target {left, right} wheel speed`` each at the same value in order to moves the robot
        straight ahead.
        This distributed controller is a simple proportional controller ``PID(5, 0, 0, max_out=16.6, min_out=-16.6)``
        that takes in input the difference between the front and back distances measured by the agent.

        :param state: object containing the agent information
        :param dt: control step duration

        """
        # Don't move the first and last robots in the line
        # if state.index == 0 or state.index == self.N - 1:
        if np.isclose(round(state.goal_position[0], 2), round(state.initial_position[0], 2), rtol=1e-2):
            return 0, 0
        else:
            speed = self.p_distributed_controller.step(self.compute_difference(state), dt)
            return speed, 0


class OmniscientController:
    """
    The robots can be moved also following an optimal “omniscient” controller. In this case, based on the poses of
    the robots, the expert controller moves the robots at a constant speed, clipped to a minimum of -16.6 and
    a maximum of 16.6, calculating the distance from the actual pose to the target one.

    :param name: name of the controller used (in this case omniscient)
    :param goal: task to perform (in this case distribute)
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

    def linear_vel(self, state, constant=10):
        """
        Compute the linear velocity as the ``signed_distance`` between the current and the goal position of the robot,
        along the current theta of the robot. The final velocity is then multiplied by a constant value.

        .. math:: velocity = constant * self.signed\_distance()

        :param state: object containing the agent information
        :param constant: constant value (default: 10, but used also values as 1 or 4)
        :return velocity: clipped linear velocity
        """
        velocity = constant * utils.signed_distance(state)

        return min(max(-16.6, velocity), 16.6)

    def perform_control(self, state, dt):
        """
        Move the robots using the omniscient controller by setting the target {left,right} wheel speed
        each at the same value in order to moves the robot straight ahead.

        :param state: object containing the agent information
        :param dt: control step duration
        :return speed, 0: the computed speed and the message to communicate, fixed to 0, used only in case of
                          ``prox_comm`` or ``all_sensors`` net input
        """

        speed = self.linear_vel(state)

        return speed, 0


class LearnedController:
    """
    The robots can be moved following a controller learned by a neural network.

    :param name: name of the controller used (in this case omniscient)
    :param goal: task to perform (in this case distribute)
    :param N: number of agents in the simulation
    :param net: network to be used by the controller
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    :param communication: states if the communication is used by the network
    :param kwargs: other arguments
    """

    def __init__(self, name, goal, N, net, net_input, communication, **kwargs):
        super().__init__(**kwargs)

        self.name = name
        self.goal = goal
        self.N = N

        self.net = net
        if self.net is None:
            raise ValueError("Value for net not provided")

        self.communication = communication
        self.net_controller = net.controller(thymio=self.N)
        self.net_input = net_input

    def perform_control(self, state, dt):
        """
        Extract the input sensing from the list of (7 or 14) proximity sensor readings, one for each sensor.
        The first 5 entries are from frontal sensors ordered from left to right.
        The last two entries are from rear sensors ordered from left to right.
        In case of all sensors the first 7 values refers to ``prox_values`` and the following 7 to ``prox_comm``.
        Each value is normalised dividing it by 1000, that is the mean value of the sensors.

        The obtained sensing is passed as input to the net and obtain the speed and the eventual communication to be transmitted.

        .. note:: Keep still the robots at the ends of the line and send for them alway 0 as message.


        :param state: object containing the agent information
        :param dt: control step duration
        """

        sensing = utils.get_input_sensing(self.net_input, state)

        if not self.communication:
            speed = float(self.net_controller(sensing)[0])
            comm = 0
        else:
            communication = utils.get_received_communication(state)
            speed, comm = self.net_controller(sensing, communication, state.index)
            speed = float(speed)

        # speed = min(max(-16.6, speed * 5), 16.6)

        if not np.isclose(round(state.goal_position[0], 2), round(state.initial_position[0], 2), rtol=1e-2):
            # convert communication into an int of between 0 and 10 bit
            if self.communication:
                comm = int(comm[state.index] * (2 ** 10))
            return speed, comm
        else:
            return 0, 0

from math import sin, cos

import numpy as np

from pid import PID


class ManualController:
    """
    The robots are moved following a “distributed controller”, that is a simple proportional controller PID,
    with P fixed to -0.01, whose goal is to align the robots by minimizing the difference between the values
    ​​recorded by the front and rear sensors.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p_distributed_controller = PID(-0.01, 0, 0, max_out=16.6, min_out=-16.6)

    def neighbors_distance(self, state):
        """
        Check if there is a robot ahead using the infrared sensor 2 (front-front).
        Check if there is a robot ahead using the infrared sensor 5 (back-left) and 6 (back-right).
        :return back, front: response values of the rear and front sensors
        """
        prox_values = state.prox_values
        front = prox_values[2]
        back = np.mean(np.array([prox_values[5], prox_values[6]]))

        return back, front

    def compute_difference(self, state):
        """
        :return: the difference between the response value of front and the rear sensor
        """
        back, front = self.neighbors_distance(state)

        # Apply a small correction to the distance measured by the rear sensors: the front sensor used is at a
        # different x coordinate from the point to which the rear sensor of the robot that follows points. this is
        # because of the curved shape of the face of the Thymio
        delta_x = 7.41150769
        x = 7.95

        # Maximum possible response values
        delta_x_m = 4505 * delta_x / 14
        x_m = 4505 * x / 14

        correction = x_m - delta_x_m

        out = front - correction - back

        return out

    def perform_control(self, state, dt):
        """
        Move the robots not to the end of the line using the distributed controller, setting the target {left,
        right} wheel speed each at the same value in order to moves the robot straight ahead. This distributed
        controller is a simple proportional controller PID(-0.01, 0, 0, max_out=16.6, min_out=-16.6) that takes in
        input the difference between the response value of front and the rear sensor. To the distance measured by the
        rear sensors is applied a small correction since the front sensor used is at a different x coordinate from
        the point to which the rear sensor of the robot that follows points. This is because of the curved shape of
        the face of the Thymio.

        The response values are actually intensities: the front correspond to the frontal center sensor and the back
        to the mean of the response values of the rear sensors.
        The final difference is computed ad follow: out = front - correction - back
        The speed are clipped to [min_out=-16.6, max_out=16.6].
        :param dt: control step duration

        """
        # Don't move the first and last robots in the line
        if state.initial_position[0] != state.goal_position[0]:
            speed = self.p_distributed_controller.step(self.compute_difference(state), dt)

            return speed


class OmniscientController:
    """
    The robots can be moved also following an optimal “omniscient” controller. In this case, based on the poses of
    the robots, the omniscient control moves the robots at a constant speed, calculating the distance from the
    actual pose to the target one.
    """

    def signed_distance(self, state):
        """
        :return: Signed distance between current and the goal position, along the current theta of the robot
        """
        a = state.position[0] * cos(state.angle) + state.position[1] * sin(state.angle)
        b = state.goal_position[0] * cos(state.angle) + state.goal_position[1] * sin(state.angle)

        return b - a

    def linear_vel(self, state, constant=4):
        """
        :param constant
        :return: clipped linear velocity
        """
        velocity = constant * self.signed_distance(state)
        return min(max(-16.6, velocity), 16.6)

    def move_to_goal(self, state):
        """
        Moves the thymio to the goal.
        :return: speed
        """
        return self.linear_vel(state)

    def perform_control(self, state, dt):
        """
        Move the robots using the omniscient controller by setting the target {left,right} wheel speed
        each at the same value in order to moves the robot straight ahead.
        The speed is computed as follow:
            velocity = constant * self.signed_distance()
        where the constant is set to 4 and the signed_distance is the distance between the current and the goal
        position of the robot, along the current theta of the robot.
        """
        speed = self.move_to_goal(state)

        return speed


class LearnedController():
    """
    The robots can be moved following a controller learned by a neural network.
    """

    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)

        self.net = net
        if self.net is not None:
            self.net_controller = net.controller()

    def perform_control(self, state, dt):
        """
        Extract the input sensing from the list of (7) proximity sensor readings, one for each sensors.
        The first 5 entries are from frontal sensors ordered from left to right.
        The last two entries are from rear sensors ordered from left to right.
        Then normalise each value of the list, by dividing it by 1000.

        Generate the output speed using the learned controller.

        Move the robots not to the end of the line using the controller, setting the target {left,right} wheel speed
        each at the same value in order to moves the robot straight ahead.
        """
        sensing = np.divide(np.array(state.prox_values), 1000).tolist()
        speed = float(self.net_controller(sensing)[0])

        if state.initial_position[0] != state.goal_position[0]:
            return speed
        else:
            return 0

import os

import matplotlib.pyplot as plt
import numpy as np
import pyenki

from controllers.pid import PID
from utils import my_plots
from utils import utils


class Thymio(pyenki.Thymio2):
    """
    Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.

    :param name: name of the agent
    :param index: index of the agents in the row
    :param p: proportional term of the PID controller
    :param i: integral term of the PID controller
    :param d: derivative term of the PID controller
    :param kwargs: other arguments

    :var initial_position: the initial position of the agent is set to None
    :var goal_position: the goal position of the agent is set to None
    :var goal_angle: the goal angle of the agent is set to None
    :var dictionary: the dictionary containing all the agent attributes is set to None
    :var colour: the colour of the agent is set to None
    :var p_distributed_controller: a simple proportional controller that returns the speed to apply
    """

    def __init__(self, name, index, p, i, d, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = name
        self.index = index

        self.initial_position = None
        self.goal_position = None
        self.goal_angle = 0

        self.dictionary = None

        self.colour = None
        self.p_distributed_controller = PID(p, i, d, max_out=16.6, min_out=-16.6)

    def get_input_sensing(self):
        """
        :return sensing: the sensing perceived by the robot based on the net input
        """

        if len(self.prox_comm_events) == 0:
            prox_comm = {'sender': {'intensities': [0, 0, 0, 0, 0, 0, 0]}}
        else:
            prox_comm = utils.get_prox_comm(self)

        prox_values = getattr(self, 'prox_values').copy()

        sensing = utils.get_all_sensors(prox_values, prox_comm)

        return sensing

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
            Compute the error as the difference between the goal and the actual position
            and use it to compute the velocity of the robot through a proportional controller.

        :param dt: control step duration
        """
        error = self.goal_position[0] - self.position[0]

        speed = self.p_distributed_controller.step(error, dt)

        self.motor_right_target = speed
        self.motor_left_target = speed


def main(directory, P, I, D):
    """

    :param directory: directory were to save the images containing the step response
    :param P: proportional term of the PID controller
    :param I: integral term of the PID controller
    :param D: derivative term of the PID controller
    """
    world = pyenki.World()

    myt = Thymio(name='myt1', index=1, p=P, i=I, d=D, use_aseba_units=False)

    myt.position = (0, 0)
    myt.initial_position = myt.position
    myt.goal_position = (20, 0)
    myt.angle = 0

    world.add_object(myt)

    dt = 0.1
    T = 8
    steps = int(T // dt)

    positions = []
    for s in range(steps):
        positions.append(myt.position[0])
        world.step(dt=dt)

    time = np.arange(0, len(positions) * dt, dt)[:len(positions)]
    plt.figure()
    plt.xlabel('time (seconds)', fontsize=11)
    plt.ylabel('x position', fontsize=11)

    plt.plot(time, positions, label='Kp = %s, Ki = %s, Kd = %s' % (round(P, 2), round(I, 2), round(D, 2)))
    plt.plot(time, [myt.goal_position[0]] * len(positions), label='Setpoint', ls='--', color='black')
    plt.title('Step response')
    plt.ylim(top=myt.goal_position[0] + 10)
    plt.xlim(0, 7)

    plt.legend()
    my_plots.save_visualisation('Step-responsep=kp%ski%skd%s' % (round(P, 2), round(I, 2), round(D, 2)), directory)


if __name__ == '__main__':
    dir_ = os.path.join('controllers', 'images')
    utils.check_dir(dir)

    Prop = 100
    Int = 0
    Der = 0

    main(dir_, Prop, Int, Der)

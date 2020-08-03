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
    """

    def __init__(self, name, index, p, i, d, **kwargs) -> None:
        """
        :param name
        :param index
        :param kwargs
        """
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
        :return sensing:
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
        Move the robots in such a way they stand at equal distances from each other.
        Enable communication and send at each timestep a message containing the index.
        It is possible to use the distributed, the omniscient or the learned controller.
        :param dt: control step duration
        """
        error = self.goal_position[0] - self.position[0]

        speed = self.p_distributed_controller.step(error, dt)
        self.motor_right_target = speed
        self.motor_left_target = speed


dir = os.path.join('controllers', 'images')
utils.check_dir(dir)

world = pyenki.World()
p = 100
i = 0
d = 0

myt = Thymio(name='myt1', index=1, p=p, i=i, d=d, use_aseba_units=False)

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
    diff = abs(myt.position[0] - myt.goal_position[0])

    world.step(dt=dt)

time = np.arange(0, len(positions) * dt, dt)[:len(positions)]
plt.figure()
plt.xlabel('time (seconds)', fontsize=11)
plt.ylabel('x position', fontsize=11)

plt.plot(time, positions, label='Kp = %s, Ki = %s, Kd = %s' % (round(p, 2), round(i, 2), round(d, 2)))
plt.plot(time, [myt.goal_position[0]] * len(positions), label='Setpoint', ls='--', color='black')
plt.title('Step response')
plt.ylim(top=myt.goal_position[0] + 10)
plt.xlim(0, 7)

plt.legend()
my_plots.save_visualisation('Step-responsep=kp%ski%skd%s' % (round(p, 2), round(i, 2), round(d, 2)), dir)

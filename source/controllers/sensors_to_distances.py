import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyenki

from utils import my_plots
from utils import utils


class Thymio(pyenki.Thymio2):
    """
    Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.

    :param name: name of the agent
    :param index: index of the agents in the row
    :param kwargs: other arguments

    :var initial_position: the initial position of the agent is set to None
    :var dictionary: the dictionary containing all the agent attributes is set to None
    :var colour: the colour of the agent is set to None
    """

    def __init__(self, name, index, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = name
        self.index = index

        self.initial_position = None

        self.dictionary = None

        self.colour = None

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
            Enable communication and send at each timestep a message.

        :param dt: control step duration
        """
        self.prox_comm_tx = 0
        self.prox_comm_enable = True


def main(distances, front_prox_values, back_prox_values, front_prox_comms, back_prox_comms,
         myt_quantity, min_distance):
    """

    :param distances: array containing the distances among the agents for each experiment
    :param front_prox_values: array containing the value of the frontal sensor using prox_values
    :param back_prox_values: array containing the value corresponding to the average of the rear sensor readings using prox_values
    :param front_prox_comms: array containing the value of the frontal sensor using prox_comm
    :param back_prox_comms: array containing the value corresponding to the average of the rear sensor readings using prox_comm
    :param myt_quantity: number of agents
    :param min_distance: length of the agents
    """

    for idx, distance in enumerate(distances):

        initial_positions = np.array([0, min_distance + distance, (min_distance + distance) * 2], dtype=np.float64)

        world = pyenki.World()

        myts = []
        for i in range(myt_quantity):
            myt = Thymio(name='myt%d' % (i + 1), index=i, use_aseba_units=False)

            myt.position = (initial_positions[i], 0)
            myt.initial_position = myt.position

            # Reset the parameters
            myt.angle = 0

            myt.prox_comm_tx = 0
            myt.prox_comm_enable = True

            myts.append(myt)
            world.add_object(myt)

        print('distance = %d' % distance)
        # for m in myts:
        #     print(m.name, round(m.initial_position[0], 1), round(m.position[0], 1))

        sensors = dict()

        a = []
        b = []
        c = []
        d = []
        for _ in range(5):
            world.step(dt=0.1)

            sensing = myts[1].get_input_sensing()
            front_prox_value = sensing[2]
            back_prox_value = np.mean([sensing[5], sensing[6]])

            front_prox_comm = sensing[9]
            back_prox_comm = np.mean([sensing[12], sensing[13]])

            a.append(int(front_prox_value))
            b.append(int(back_prox_value))
            c.append(int(front_prox_comm))
            d.append(int(back_prox_comm))

        front_prox_values[idx] = int(np.mean(a))
        back_prox_values[idx] = int(np.mean(b))
        front_prox_comms[idx] = int(np.mean(c))
        back_prox_comms[idx] = int(np.mean(d))

        sensors['front_prox_values'] = int(np.mean(a))
        sensors['back_prox_values'] = int(np.mean(b))
        sensors['front_prox_comm'] = int(np.mean(c))
        sensors['back_prox_comm'] = int(np.mean(d))

        print(sensors)

    sensing_to_distances = [distances, front_prox_values, back_prox_values, front_prox_comms, back_prox_comms]

    file = os.path.join('controllers', 'sensing_to_distances.pkl')
    with open(file, 'wb') as f:
        pickle.dump(sensing_to_distances, f)

    plt.figure()
    plt.xlabel('distance', fontsize=11)
    plt.ylabel('sensing', fontsize=11)

    plt.plot(distances, front_prox_values, label='front_prox_values')
    plt.plot(distances, back_prox_values, label='back_prox_values')
    plt.plot(distances, front_prox_comms, label='front_prox_comm')
    plt.plot(distances, back_prox_comms, label='back_prox_comm')
    plt.legend()

    dir = os.path.join('controllers', 'images')
    utils.check_dir(dir)

    my_plots.save_visualisation('sensing_to_distances', dir)

    # world.run_in_viewer(cam_position=(10.9, 0), cam_altitude=50.0, cam_yaw=0.0, cam_pitch=-np.pi / 2,
    #                     walls_height=10.0, orthographic=True, period=0.1)


if __name__ == '__main__':
    distances = np.arange(48)
    front_prox_values = np.zeros(48)
    back_prox_values = np.zeros(48)
    front_prox_comms = np.zeros(48)
    back_prox_comms = np.zeros(48)

    myt_quantity = 3
    # min_distance = 10.9
    min_distance = 11

    main(distances, front_prox_values, back_prox_values, front_prox_comms, back_prox_comms,
         myt_quantity, min_distance)

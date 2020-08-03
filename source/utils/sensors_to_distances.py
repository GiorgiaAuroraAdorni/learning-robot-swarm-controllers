import os
import pickle

import matplotlib.pyplot as plt
import my_plots
import numpy as np
import pyenki

import utils


class Thymio(pyenki.Thymio2):
    """
    Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
    """

    def __init__(self, name, index, **kwargs) -> None:
        """
        :param name
        :param index
        :param kwargs
        """
        super().__init__(**kwargs)

        self.name = name
        self.index = index

        self.initial_position = None
        self.goal_angle = 0

        self.dictionary = None

        self.colour = None

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
        self.prox_comm_tx = 0
        self.prox_comm_enable = True

distances = np.arange(48)
front_prox_values = np.zeros(48)
back_prox_values = np.zeros(48)
front_prox_comms = np.zeros(48)
back_prox_comms = np.zeros(48)

for idx, distance in enumerate(distances):

    myt_quantity = 3
    # min_distance = 10.9
    min_distance = 11

    initial_positions = np.array([0, min_distance + distance, (min_distance + distance) * 2 ], dtype=np.float64)

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

# front_prox_values[front_prox_values == 0] = np.nan
# back_prox_values[back_prox_values == 0] = np.nan
# front_prox_comms[front_prox_comms == 0] = np.nan
# back_prox_comms[back_prox_comms == 0] = np.nan

sensing_to_distances = [distances, front_prox_values, back_prox_values, front_prox_comms, back_prox_comms]

file = os.path.join('controllers', 'sensing_to_distances.pkl')
with open(file, 'wb') as f:
    pickle.dump(sensing_to_distances, f)


plt.figure()
plt.xlabel('distance', fontsize=11)
plt.ylabel('sensing', fontsize=11)

plt.plot(distances, front_prox_values, label='front_prox_values')
plt.plot(distances, back_prox_values,  label='back_prox_values')
plt.plot(distances, front_prox_comms,  label='front_prox_comm')
plt.plot(distances, back_prox_comms,   label='back_prox_comm')
plt.legend()

my_plots.save_visualisation('sensing_to_distances', 'controllers')

# world.run_in_viewer(cam_position=(10.9, 0), cam_altitude=50.0, cam_yaw=0.0, cam_pitch=-np.pi / 2,
#                     walls_height=10.0, orthographic=True, period=0.1)

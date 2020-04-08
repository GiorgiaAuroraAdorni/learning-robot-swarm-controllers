import json
import os
import pickle
import sys
from math import pi, sqrt, sin, cos

import numpy as np
import pyenki
from utils import check_dir


# Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
class DistributedThymio2(pyenki.Thymio2):

    def __init__(self, name, index, goal_position, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.index = index
        self.goal_position = goal_position
        self.dictionary = dict()

    def euclidean_distance(self):
        """
        :return: Euclidean distance between current and the goal position
        """
        return sqrt(pow((self.goal_position[0] - self.position[0]), 2) +
                    pow((self.goal_position[1] - self.position[1]), 2))

    def signed_distance(self):
        """
        :return: Signed distance between current and the goal position, along the current theta of the robot
        """

        a = self.position[0] * cos(self.angle) + self.position[1] * sin(self.angle)
        b = self.goal_position[0] * cos(self.angle) + self.goal_position[1] * sin(self.angle)

        return b - a

    def linear_vel(self, constant=4):
        """
        :param constant
        :return: clipped linear velocity
        """
        # velocity = constant * self.euclidean_distance()
        velocity = constant * self.signed_distance()
        return min(max(-500, velocity), 500)

    def move_to_goal(self):
        """
        Moves the thymio to the goal.
        :return: speed
        """
        return self.linear_vel()

    def generate_dict(self, speed):
        """

        :param speed:
        """
        prox_comm_events_intensities = ''
        prox_comm_events_payloads = ''
        prox_comm_events_message = ''

        if len(self.prox_comm_events) > 0:
            prox_comm_events_intensities = self.prox_comm_events[0].intensities
            prox_comm_events_payloads = self.prox_comm_events[0].payloads
            prox_comm_events_message = self.prox_comm_events[0].rx

        self.dictionary = {
            'name': self.name,
            'index': self.index,
            'prox_values': self.prox_values,
            'prox_comm_events_intensities': prox_comm_events_intensities,
            'prox_comm_events_payloads': prox_comm_events_payloads,
            'prox_comm_events_message': prox_comm_events_message,
            'position': self.position,
            'goal_position': self.goal_position,
            'speed': speed
        }

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step: Move the robots in such a way they stand at equal distances from each other without
        using communication.
        :param dt: control step duration
        """
        self.prox_comm_enable = True
        self.prox_comm_tx = self.index

        speed = self.move_to_goal()
        self.motor_left_target = speed
        self.motor_right_target = speed

        self.generate_dict(speed)


def setup(myt_quantity, aseba: bool = False):
    """
    Set up the world and create the thymios
    :param aseba
    :return world
    """
    # Create an unbounded world
    world = pyenki.World()

    # Create multiple Thymios and position them such as all x-axes are aligned
    myts = [DistributedThymio2(name='myt%d' % (i + 1), index=i, goal_position=None, use_aseba_units=aseba)
            for i in range(myt_quantity)]

    # The robots are already arranged in an "indian row" (all x-axes aligned) and within the proximity sensor range
    # ~ 14 cm is the proximity sensors maximal range
    distances = np.random.randint(5, 10, 8)

    # Distance between the origin and the front of the robot along x-axis
    constant = 7.95

    for i, myt in enumerate(myts):
        if i == 0:
            prev_pos = 0
        else:
            prev_pos = myts[i - 1].position[0]

        current_pos = prev_pos + float(distances[i]) + constant
        myt.position = (current_pos, 0)
        myt.angle = 0

    # Decide the goal pose for each robot
    goal_positions = np.linspace(myts[0].position[0], myts[myt_quantity - 1].position[0], num=8)

    for i, myt in enumerate(myts):
        myt.goal_position = (goal_positions[i], 0)
        world.add_object(myt)

    return world, myts


def run(simulation, myts, world: pyenki.World, gui: bool = False, T: float = 10, dt: float = 0.1) -> None:
    """
    :param file
    :param world
    :param gui
    :param T
    :param dt: update timestep in seconds, should be below 1 (typically .02-.1)
    """

    if gui:
        # We can either run a simulation [in real-time] inside a Qt application
        world.run_in_viewer(cam_position=(60, 0), cam_altitude=110.0, cam_yaw=0.0, cam_pitch=-pi / 2,
                            walls_height=10.0, orthographic=True, period=0.1)
    else:
        # Run the simulation as fast as possible
        steps = int(T // dt)

        data = []
        iteration = []

        for s in range(steps):
            if s > 0:
                for el in myts:
                    iteration.append(el.dictionary)

                    if len(iteration) == myt_quantity:
                        data.append(iteration)
                        iteration = []

            world.step(dt)

        out_dir = 'out/'
        check_dir(out_dir)
        pkl_file = os.path.join(out_dir, 'simulation-%d.pkl' % simulation)
        json_file = os.path.join(out_dir, 'simulation-%d.json' % simulation)

        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    myt_quantity = 8
    world, myts = setup(myt_quantity)

    simulation = 1

    try:
        run(simulation, myts, world, '--gui' in sys.argv)
    except:
        raise

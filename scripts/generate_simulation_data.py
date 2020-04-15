import json
import os
import pickle
import sys
from math import pi, sin, cos

import numpy as np
import pyenki
from tqdm import tqdm
from utils import check_dir


# Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
class DistributedThymio2(pyenki.Thymio2):

    def __init__(self, name, index, initial_position, goal_position, goal_angle, dictionary, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.index = index
        self.initial_position = initial_position
        self.goal_position = goal_position
        self.goal_angle = goal_angle
        self.dictionary = dictionary

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
        velocity = constant * self.signed_distance()
        return min(max(-500, velocity), 500)

    def move_to_goal(self):
        """
        Moves the thymio to the goal.
        :return: speed
        """
        return self.linear_vel()

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
        Move the robots in such a way they stand at equal distances from each other using the omniscient controller.
        :param dt: control step duration
        """
        self.prox_comm_enable = True
        self.prox_comm_tx = self.index

        speed = self.move_to_goal()
        self.motor_left_target = speed
        self.motor_right_target = speed


def init_positions(myts):
    """

    :param myts:
    """
    myt_quantity = len(myts)

    # The minimum distance between two Thymio [wheel - wheel] is 12 cm
    min_distance = 10.9  # in the previous version it was set at 7.95
    avg_gap = 12  # can vary in the range [6, 14]

    # The robots are already arranged in an "indian row" (all x-axes aligned) and within the proximity sensor range
    # ~ 14 cm is the proximity sensors maximal range
    maximum_gap = 14
    std = 8

    first_x = 0
    last_x = (min_distance + avg_gap) * (myt_quantity - 1)

    distances = min_distance + np.clip(np.random.normal(avg_gap, std, myt_quantity - 1), 1, maximum_gap)
    distances = distances / np.sum(distances) * last_x
    # distances = np.random.randint(5, 10, 8)  # in the previous version

    for i, myt in enumerate(myts):
        # Position the first and last robot at a fixed distance
        if i == 0:
            myt.position = (first_x, 0)
        elif i == myt_quantity:
            myt.position = (last_x, 0)
        else:
            prev_pos = myts[i - 1].position[0]
            # current_pos = prev_pos + float(distances[i]) + constant  # in the previous version
            current_pos = prev_pos + distances[i - 1]
            myt.position = (current_pos, 0)

        myt.angle = 0
        myt.initial_position = myt.position
        #  Reset the dictionary
        myt.dictionary = None

    # Decide the goal pose for each robot
    goal_positions = np.linspace(myts[0].position[0], myts[myt_quantity - 1].position[0], num=myt_quantity)

    for i, myt in enumerate(myts):
        myt.goal_position = (goal_positions[i], 0)


def setup(myt_quantity, aseba: bool = False):
    """
    Set up the world and create the thymios
    :param myt_quantity: number of robot in the simulation
    :param aseba
    :return world, myts
    """
    # Create an unbounded world
    world = pyenki.World()

    # Create multiple Thymios and position them such as all x-axes are aligned
    myts = [DistributedThymio2(name='myt%d' % (i + 1), index=i, initial_position=None, goal_position=None,
                               goal_angle=0, dictionary=None, use_aseba_units=aseba) for i in range(myt_quantity)]

    for myt in myts:
        world.add_object(myt)

    return world, myts


def get_prox_comm(myt):
    """
    :param myt
    :return prox_comm: dictionary of dictionaries with the name of the sender as key and the intensities and the
    message as values
    """
    prox_comm = {}

    # Extract sender and intensities
    prox_comm_events = myt.prox_comm_events

    if len(prox_comm_events) > 0:
        for idx, _ in enumerate(prox_comm_events):
            sender = prox_comm_events[idx].rx + 1
            intensities = prox_comm_events[idx].intensities

            prox_comm['myt%d' % sender] = {'intensities': intensities}

    return prox_comm


def generate_dict(myt):
    """
    Save data in a dictionary
    :param myt
    :return dictionary
    """
    myt.dictionary = {
        'name': myt.name,
        'index': myt.index,
        'prox_values': myt.prox_values,
        'prox_comm': get_prox_comm(myt),
        'initial_position': myt.initial_position,
        'position': myt.position,
        'angle': myt.angle,
        'goal_position': myt.goal_position,
        'goal_angle': myt.goal_angle,
        'motor_left_target': myt.motor_left_target,
        'motor_right_target': myt.motor_right_target
    }

    dictionary = myt.dictionary.copy()
    return dictionary


def update_dict(myt):
    """
    Updated data in the dictionary instead of rewrite every field to optimise performances
    :param myt
    :return dictionary
    """
    myt.dictionary['prox_values'] = myt.prox_values
    myt.dictionary['prox_comm'] = get_prox_comm(myt)
    myt.dictionary['position'] = myt.position
    myt.dictionary['angle'] = myt.angle
    myt.dictionary['motor_left_target'] = myt.motor_left_target
    myt.dictionary['motor_right_target'] = myt.motor_right_target

    dictionary = myt.dictionary.copy()

    return dictionary


def run(simulation, myts, world: pyenki.World, gui: bool = False, T: float = 100, dt: float = 0.1,
        tol: float = 0.1) -> None:
    """
    :param simulation
    :param myts
    :param world
    :param gui
    :param T
    :param dt: update timestep in seconds, should be below 1 (typically .02-.1)
    :param tol: tolerance used to verify if the robot reaches the target
    """
    myt_quantity = len(myts)

    if gui:
        # We can either run a simulation [in real-time] inside a Qt application
        world.run_in_viewer(cam_position=(60, 0), cam_altitude=150.0, cam_yaw=0.0, cam_pitch=-pi / 2,
                            walls_height=10.0, orthographic=True, period=0.1)
    else:
        # Run the simulation as fast as possible
        steps = int(T // dt)

        data = []
        iteration = []

        stop_iteration = False

        for s in range(steps):
            counter = 0
            if stop_iteration:
                break

            if s > 0:
                for i, myt in enumerate(myts):
                    if myt.dictionary is None:
                        dictionary = generate_dict(myt)
                    else:
                        dictionary = update_dict(myt)
                    iteration.append(dictionary)

                    # Check if the robot has reached the target
                    diff = abs(iteration[i]['position'][0] - iteration[i]['goal_position'][0])
                    if diff < tol:
                        counter += 1

            # Check is the step is finished
            if len(iteration) == myt_quantity:
                data.append(iteration)
                iteration = []

            # Stop the simulation if all the robots have reached the goal
            if counter == myt_quantity:
                stop_iteration = True
            else:
                world.step(dt)

        out_dir = 'out/%dmyts/' % myt_quantity
        # os.makedirs(out_dir, exist_ok=True)
        check_dir(out_dir)
        pkl_file = os.path.join(out_dir, 'simulation-%d.pkl' % simulation)
        json_file = os.path.join(out_dir, 'simulation-%d.json' % simulation)

        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    simulations = 1000
    myt_quantity = 5
    world, myts = setup(myt_quantity)

    for simulation in tqdm(range(simulations)):
        try:
            init_positions(myts)
            run(simulation, myts, world, '--gui' in sys.argv)
        except Exception as e:
            print('ERROR: ', e)

import json
import os
import pickle
import sys
from math import pi, sin, cos

import numpy as np
import pyenki
import torch
from tqdm import tqdm
from pid import PID
from utils import check_dir, visualise_simulation, visualise_simulations_comparison


# Superclass: `pyenki.Thymio2` -> the world update step will automatically call the Thymio `controlStep`.
class DistributedThymio2(pyenki.Thymio2):

    def __init__(self, controller, name, index, initial_position, goal_position, goal_angle, dictionary, net,
                 **kwargs) -> None:
        """

        :param controller:
        :param name:
        :param index:
        :param initial_position:
        :param goal_position:
        :param goal_angle:
        :param dictionary:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.name = name
        self.index = index
        self.initial_position = initial_position
        self.goal_position = goal_position
        self.goal_angle = goal_angle
        self.distribute = True

        #  Encodes speed between -16.6 and +16.6 (aseba units: [-500,500])
        self.controller = controller
        self.p_distributed_controller = PID(-0.01, 0, 0, max_out=16.6, min_out=-16.6)

        self.dictionary = dictionary
        self.net = net
        if self.net is not None:
            self.net_controller = net.controller()

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
        return min(max(-16.6, velocity), 16.6)

    def move_to_goal(self):
        """
        Moves the thymio to the goal.
        :return: speed
        """
        return self.linear_vel()

    def neighbors_distance(self):
        """
        Check if there is a robot ahead using the infrared sensor 2 (front-front).
        Check if there is a robot ahead using the infrared sensor 5 (back-left) and 6 (back-right).
        :return back, front: response values of the rear and front sensors
        """
        prox_values = self.prox_values
        front = prox_values[2]
        back = np.mean(np.array([prox_values[5], prox_values[6]]))

        return back, front

    def compute_difference(self):
        """

        :return: the difference between the response value of front and the rear sensor
        """
        back, front = self.neighbors_distance()

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

    def distributed_controller(self, dt):
        """
        :param dt:

        """
        # Don't move the first and last robots in the line
        if self.initial_position[0] != self.goal_position[0]:
            speed = self.p_distributed_controller.step(self.compute_difference(), dt)

            self.motor_left_target = speed
            self.motor_right_target = speed

    def omniscient_controller(self):
        """

        """
        speed = self.move_to_goal()

        self.motor_left_target = speed
        self.motor_right_target = speed

    def learned_controller(self):
        """

        """
        sensing = np.divide(np.array(self.prox_values), 1000).tolist()
        speed = float(self.net_controller(sensing)[0])

        if self.initial_position[0] != self.goal_position[0]:
            self.motor_left_target = speed
            self.motor_right_target = speed

    def controlStep(self, dt: float) -> None:
        """
        Perform one control step:
        Move the robots in such a way they stand at equal distances from each other using the omniscient controller.
        :param dt: control step duration
        """
        self.prox_comm_enable = True
        self.prox_comm_tx = self.index

        if self.controller == 'distributed':
            self.distributed_controller(dt)
        elif self.controller == 'omniscient':
            self.omniscient_controller()
        elif self.controller == 'net1':
            self.learned_controller()


def init_positions(myts, variate_pose=False, min_distance = 10.9, avg_gap = 8, maximum_gap = 14, x=None):
    """
    myts, simulation, variate_pose
    :param myts:
    """
    myt_quantity = len(myts)

    # The minimum distance between two Thymio [wheel - wheel] is 12 cm
    # min_distance = 10.9  # in the previous version it was set at 7.95
    # avg_gap = 8  # 12  # can vary in the range [6, 14]

    # The robots are already arranged in an "indian row" (all x-axes aligned) and within the proximity sensor range
    # ~ 14 cm is the proximity sensors maximal range
    # maximum_gap = 14  # 12
    std = 8

    first_x = 0
    last_x = (min_distance + avg_gap) * (myt_quantity - 1)

    if variate_pose:
        distances = [min_distance + x]
    else:
        distances = min_distance + np.clip(np.random.normal(avg_gap, std, myt_quantity - 1), 1, maximum_gap)
        distances = distances / np.sum(distances) * last_x
    # distances = np.random.randint(5, 10, 8)  # in the previous version

    # Decide the goal pose for each robot
    goal_positions = np.linspace(first_x, last_x, num=myt_quantity)

    for i, myt in enumerate(myts):
        # Position the first and last robot at a fixed distance
        if i == 0:
            myt.position = (first_x, 0)
        elif i == myt_quantity - 1:
            myt.position = (last_x, 0)
        else:
            prev_pos = myts[i - 1].position[0]
            # current_pos = prev_pos + float(distances[i]) + constant  # in the previous version
            current_pos = prev_pos + distances[i - 1]
            myt.position = (current_pos, 0)
            # myt.position = (goal_positions[i], 0)

        myt.angle = 0
        myt.initial_position = myt.position

        #  Reset the dictionary
        myt.dictionary = None

        myt.goal_position = (goal_positions[i], 0)


def setup(controller, myt_quantity, model_dir, aseba: bool = False):
    """
    Set up the world and create the thymios
    :param controller
    :param myt_quantity: number of robot in the simulation
    :param model_dir
    :param aseba
    :return world, myts
    """
    # Create an unbounded world
    world = pyenki.World()

    if model_dir is not None:
        net = torch.load('%s/%s' % (model_dir, controller))
    else:
        net = None

    # Create multiple Thymios and position them such as all x-axes are aligned
    myts = [DistributedThymio2(controller=controller,
                               name='myt%d' % (i + 1),
                               index=i,
                               initial_position=None,
                               goal_position=None,
                               goal_angle=0,
                               dictionary=None,
                               net=net,
                               use_aseba_units=aseba)
            for i in range(myt_quantity)]

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


def run(simulation, myts, runs_dir,
        world: pyenki.World, gui: bool = False, T: float = 5, dt: float = 0.1, tol: float = 0.1) -> None:
    """
    :param simulation
    :param myts
    :param runs_dir
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

        complete_data = []
        complete_iteration = []

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

                    # Do not include the 2 thymio at the ends, but only the ones that have to move.
                    if i != 0 and i != (myt_quantity - 1):
                        iteration.append(dictionary)

                        # Check if the robot has reached the target
                        diff = abs(dictionary['position'][0] - dictionary['goal_position'][0])
                        if diff < tol:
                            counter += 1

                    complete_iteration.append(dictionary)

            # Check is the step is finished
            if len(iteration) == myt_quantity - 2:
                data.append(iteration)
                iteration = []

            if len(complete_iteration) == myt_quantity:
                complete_data.append(complete_iteration)
                complete_iteration = []

            # Stop the simulation if all the robots have reached the goal
            if counter == myt_quantity - 2:
                stop_iteration = True
            else:
                world.step(dt)

        pkl_file = os.path.join(runs_dir, 'simulation-%d.pkl' % simulation)
        json_file = os.path.join(runs_dir, 'simulation-%d.json' % simulation)
        c_pkl_file = os.path.join(runs_dir, 'complete-simulation-%d.pkl' % simulation)
        # c_json_file = os.path.join(runs_dir, 'complete-simulation-%d.json' % simulation)

        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

        # with open(json_file, 'w', encoding='utf-8') as f:
        #     json.dump(data, f, ensure_ascii=False, indent=4)

        with open(c_pkl_file, 'wb') as f:
            pickle.dump(complete_data, f)
        #
        # with open(c_json_file, 'w', encoding='utf-8') as f:
        #     json.dump(complete_data, f, ensure_ascii=False, indent=4)


def generate__simulation(runs_dir, simulations, controller, myt_quantity, model_dir=None):
    """

    :param runs_dir:
    :param model_dir
    :param simulations:
    :param controller:
    :param myt_quantity:
    """

    world, myts = setup(controller, myt_quantity, model_dir)

    for simulation in tqdm(range(simulations)):
        try:
            init_positions(myts)
            run(simulation, myts, runs_dir, world, '--gui' in sys.argv)
        except Exception as e:
            print('ERROR: ', e)


if __name__ == '__main__':
    myt_quantity = 5

    model = 'net1'
    controller = model  # 'distributed'  # 'omniscient'

    dataset = '%dmyts-%s' % (myt_quantity, controller)

    runs_dir = os.path.join('datasets/', dataset)
    model_dir = 'models/distributed/%s' % model

    check_dir(runs_dir)
    check_dir(model_dir)

    # generate__simulation(runs_dir, simulations=1000, controller=controller, myt_quantity=myt_quantity)
    generate__simulation(runs_dir, simulations=1000, controller=controller, myt_quantity=myt_quantity, model_dir=model_dir)
    img_dir = '%s/images/' % runs_dir

    # visualise_simulation(runs_dir, img_dir, 0, 'Distribution simulation %d - %s controller' % (0, controller))
    # visualise_simulation(runs_dir, img_dir, 1, 'Distribution simulation %d - %s controller' % (1, controller))
    # visualise_simulation(runs_dir, img_dir, 2, 'Distribution simulation %d - %s controller' % (2, controller))
    # visualise_simulation(runs_dir, img_dir, 3, 'Distribution simulation %d - %s controller' % (3, controller))
    # visualise_simulation(runs_dir, img_dir, 4, 'Distribution simulation %d - %s controller' % (4, controller))
    # visualise_simulations_comparison(runs_dir, img_dir, 'Distribution of all simulations - %s controller' % controller)

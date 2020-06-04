import json
import os
import pickle
import re
from math import pi

import numpy as np
import pandas as pd
import pyenki
import torch
from tqdm import tqdm

from controllers import distributed_controllers
from distributed_thymio import DistributedThymio2
from my_plots import my_scatterplot
from utils import extract_input_output, get_prox_comm


class GenerateSimulationData:
    MANUAL_CONTROLLER = "manual"
    OMNISCIENT_CONTROLLER = "omniscient"
    LEARNED_CONTROLLER = r"^learned"

    @classmethod
    def setup(cls, controller_factory, myt_quantity, aseba: bool = False):
        """
        Set up the world as an unbounded world.
        :param controller_factory: if the controller is passed, load the learned network
        :param myt_quantity: number of robot in the simulation
        :param aseba
        :return world, myts
        """
        # Create an unbounded world
        world = pyenki.World()

        myts = []

        for i in range(myt_quantity):
            controller = controller_factory()
            myt = DistributedThymio2(name='myt%d' % (i + 1), index=i, controller=controller, use_aseba_units=aseba)

            myts.append(myt)
            world.add_object(myt)

        return world, myts

    @classmethod
    def init_positions(cls, myts, net_input, avg_gap, variate_pose=False, min_distance=10.9, x=None):
        """
        Create multiple Thymios and position them such as all x-axes are aligned.
        The robots are already arranged in an "indian row" (all x-axes aligned) and within the proximity sensor range.
        The distance between the first and the last robot is computed in this way:
        (min_distance + avg_gap) * (myt_quantity - 1).
        The distances among the robot are computed by drawing (myt_quantity-1) real random gaps in using a normal
        distribution with mean equal to the average gap and stdev fixed to 8. This values are clipped between 1 and the
        maximum_gap. Then, the minimum distance is added to all the distances, in order to move from the ICR to the front of
        the thymio. The distances obtained are rescaled in such a way their sum corresponds to the total gap that is known.
        :param myts
        :param net_input
        :param avg_gap: for the prox_values the default value is 8cm; for the prox_comm the default value is 25cm
        :param variate_pose
        :param min_distance: the minimum distance between two Thymio [wheel - wheel] is 10.9 cm.
        :param x
        """
        myt_quantity = len(myts)
        std = 8

        # maximum_gap: corresponds to the proximity sensors maximal range and is 14cm for the prox_values and
        # 50cm for prox_comm
        if net_input == 'prox_values':
            maximum_gap = 14
        elif net_input == 'prox_comm':
            maximum_gap = 50
        elif net_input == 'all_sensors':
            maximum_gap = avg_gap * 2
        else:
            raise ValueError("Invalid value for net_input")

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

            myt.initial_position = myt.position

            #  Reset the dictionary
            myt.dictionary = None
            myt.angle = 0

            myt.goal_position = (goal_positions[i], 0)

    @classmethod
    def generate_dict(cls, myt):
        """
        Save data in a dictionary FIXME rename in state_dict
        :param myt
        :return dictionary:
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

    @classmethod
    def update_dict(cls, myt):
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

    @classmethod
    def save_simulation(cls, complete_data, data, runs_dir):
        """

        :param complete_data:
        :param data:
        :param runs_dir:
        :return:
        """
        pkl_file = os.path.join(runs_dir, 'simulation.pkl')
        json_file = os.path.join(runs_dir, 'simulation.json')

        c_pkl_file = os.path.join(runs_dir, 'complete-simulation.pkl')
        c_json_file = os.path.join(runs_dir, 'complete-simulation.json')

        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        with open(c_pkl_file, 'wb') as f:
            pickle.dump(complete_data, f)

        with open(c_json_file, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, ensure_ascii=False, indent=4)

    @classmethod
    def run(cls, myts, runs, complete_runs,
            world: pyenki.World, gui: bool = False, T: float = 2, dt: float = 0.1, tol: float = 0.1) -> None:
        """
        Run the simulation as fast as possible or using the real time GUI.
        Generate two different type of simulation data, one with all the thymios and the other without including the 2
        thymios at the ends, but only the ones that have to move.
        If all the robots have reached their target, stop the simulation.
        :param myts
        :param runs
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
                            dictionary = cls.generate_dict(myt)
                        else:
                            dictionary = cls.update_dict(myt)

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

            runs.append(data)
            complete_runs.append(complete_data)

    @classmethod
    def generate_simulation(cls, run_dir, n_simulations, controller, myt_quantity, args, model_dir=None,
                            model=None):
        """

        :param run_dir:
        :param n_simulations:
        :param controller:
        :param model_dir:
        :param myt_quantity:
        :param args:
        :param model:
        """
        if controller == cls.MANUAL_CONTROLLER:
            def controller_factory(**kwargs):
                return distributed_controllers.ManualController(net_input=args.net_input, **kwargs)

        elif controller == cls.OMNISCIENT_CONTROLLER:
            controller_factory = distributed_controllers.OmniscientController
        elif re.match(cls.LEARNED_CONTROLLER, controller):
            # controller_factory = lambda **kwargs: d_c.LearnedController(net=net, **kwargs)
            net = torch.load('%s/%s' % (model_dir, model))

            def controller_factory(**kwargs):
                return distributed_controllers.LearnedController(net=net, net_input=args.net_input, **kwargs)
        else:
            raise ValueError("Invalid value for controller")

        world, myts = cls.setup(controller_factory, myt_quantity)

        runs = []
        complete_runs = []
        for _ in tqdm(range(n_simulations)):
            try:
                cls.init_positions(myts, args.net_input, args.avg_gap)
                cls.run(myts, runs, complete_runs, world, args.gui)
            except Exception as e:
                print('ERROR: ', e)

        cls.save_simulation(complete_runs, runs, run_dir)

    @classmethod
    def check_dataset_conformity(cls, runs_dir, runs_img, dataset, net_input):
        """
        Generate a scatter plot to check the conformity of the dataset. The plot will show the distribution of the input
        sensing, in particular, as the difference between the front sensor and the mean of the rear sensors,
        with respect to the output control of the datasets.
        :param runs_dir: directory containing the simulation
        :param runs_img: directory containing the simulation images
        :param dataset
        :param net_input
        """
        input_ = []
        output_ = []

        pickle_file = os.path.join(runs_dir, 'simulation.pkl')
        runs = pd.read_pickle(pickle_file)

        for run in runs:
            extract_input_output(run, input_, output_, net_input, 'motor_left_target')

        #  Generate a scatter plot to check the conformity of the dataset
        title = 'Dataset %s' % dataset
        file_name = 'dataset-scatterplot-%s' % dataset

        # runs_img = os.path.join(runs_img, net_input)
        # check_dir(runs_img)

        x = np.array(input_)[:, 2] - np.mean(np.array(input_)[:, 5:], axis=1)  # x: front sensor - mean(rear sensors)
        y = np.array(output_).flatten()  # y: speed
        x_label = 'sensing (%s)' % net_input
        y_label = 'control'

        my_scatterplot(x, y, x_label, y_label, runs_img, title, file_name)

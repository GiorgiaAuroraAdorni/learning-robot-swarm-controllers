import json
import os
import pickle

import numpy as np
import pyenki
import torch
from tqdm import tqdm

from thymio import DistributedThymio2
from utils import my_plots
from utils import utils


class GenerateSimulationData:
    """

    """
    MANUAL_CONTROLLER = "manual"
    OMNISCIENT_CONTROLLER = "omniscient"
    LEARNED_CONTROLLER = "learned"

    @classmethod
    def setup(cls, controller_factory, myt_quantity, aseba: bool = False):
        """
        Set up the world as an unbounded world.

        :param controller_factory: if the controller is passed, load the learned network
        :param myt_quantity: number of robot in the simulation
        :param aseba: True
        :return world, myts: world and the agents in the world
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
    def init_positions(cls, myts, net_input, avg_gap, variate_pose=False, min_distance=10.9, extension=False, x=None, epsilon=None):
        """
        Given the agents, position them such as all x-axes are aligned.
        Arrange the robots in a "single file" (all x-axes aligned) and within the proximity sensor range
        (when running the extension of task 1 it is not mandatory that the robot are within the proximity sensor range,
        since a random average gap is used).

        The distance between the first and the last robot is usually computed as follow:

        .. math:: (min\_distance + avg\_gap) * (myt\_quantity - 1)

        In case of the extended task this quantity it is not initialised.

        Usually, the distances among the robots are computed by drawing (myt_quantity - 1) real random gaps in using a normal
        distribution with mean equal to the average gap and stdev fixed to 8. This values are clipped between 1 and the
        ``maximum_gap``. Then, the ``minimum_distance`` is added to all the distances, in order to move from the ICR
        to the front of the thymio. The distances obtained are rescaled in such a way their sum corresponds to the
        total gap that is known.
        In case of the extended task the distances are obtained using the random average gap fixed and generating
        random positions in the maximum range allowed.

        :param myts: list containing all the agents and associated attributes
        :param net_input: input of the net between prox_values, prox_comm or all_sensors
        :param avg_gap: for the prox_values the default value is 8cm; for the prox_comm the default value is 25cm.
                        Can be also a random value, between 5 and 35, in case of the extended task.
        :param variate_pose: False by default, True only during the evaluation of the models when only one agent is used.
        :param min_distance: the minimum distance between two Thymio [wheel - wheel] is 10.9 cm,
                             that corresponds to the length of the agents.
        :param extension: True if is running task1_extension
        :param x: quantity of which variate the position along the x axis of the robot
                  (only during the model evaluation with variate_pose True)
        """
        myt_quantity = len(myts)
        std = 8

        # maximum_gap: corresponds to the proximity sensors maximal range and is 14cm for the prox_values and
        # 50cm for prox_comm
        if net_input == 'prox_values':
            maximum_gap = 14
        elif net_input == 'prox_comm':
            maximum_gap = 50
        elif net_input == 'all_sensors' or net_input is None:
            maximum_gap = avg_gap * 2
        else:
            raise ValueError("Invalid value for net_input")

        first_x = 0
        last_x = (min_distance + avg_gap) * (myt_quantity - 1)

        if not extension:
            if variate_pose:
                initial_positions = np.array([first_x, min_distance + x, last_x], dtype=np.float64)
                last_x = initial_positions[-1]
            else:
                distances = min_distance + np.clip(np.random.normal(avg_gap, std, myt_quantity - 1), 1, maximum_gap)
                distances = distances / np.sum(distances) * last_x

                initial_positions = np.zeros(myt_quantity)
                initial_positions[1:] = np.cumsum(distances)
        else:
            initial_positions = np.zeros(myt_quantity)

            gaps = np.random.uniform(first_x, maximum_gap, myt_quantity - 1)

            distances = np.round(gaps + min_distance, 2)
            initial_positions[1:] = np.cumsum(distances)

            last_x = initial_positions[-1]

            if distances[distances < min_distance]:
                print(distances)
                raise ValueError("Invalid initial positions")

        # Decide the goal pose for each robot
        goal_positions = np.linspace(first_x, last_x, num=myt_quantity)

        for i, myt in enumerate(myts):
            # Position the first and last robot at a fixed distance
            if variate_pose:
                myt.position = (initial_positions[i] + epsilon, epsilon)
                myt.angle = epsilon
            else:
                myt.position = (initial_positions[i], 0)
                myt.angle = 0

            myt.initial_position = myt.position

            #  Reset the parameters
            myt.dictionary = None
            myt.goal_position = (goal_positions[i], 0)

            if myt.colour is not None:
                myt.colour = None

            myt.prox_comm_tx = 0
            myt.prox_comm_enable = True

    @classmethod
    def generate_dict(cls, myt, n_sim, s, comm=None):
        """
        Save all the information of the agent in a dictionary.

        :param myt: agent
        :param n_sim: index of the simulation
        :param s: index of the timestep
        :param comm: boolean, True if the dataset is generated using the learned model with communication
        :return dictionary:
                            * 'run': n_sim,
                            * 'timestep': s,
                            * 'name': myt.name,
                            * 'index': myt.index,
                            * 'myt_quantity': myt.controller.N,
                            * 'net_input': myt.controller.net_input,
                            * 'prox_values': prox_values,
                            * 'prox_comm': utils.parse_prox_comm(prox_comm),
                            * 'all_sensors': all_sensors,
                            * 'initial_position': myt.initial_position,
                            * 'position': myt.position,
                            * 'angle': myt.angle,
                            * 'goal_position': myt.goal_position,
                            * 'goal_angle': myt.goal_angle,
                            * 'motor_left_target': myt.motor_left_target,
                            * 'motor_right_target': myt.motor_right_target,
                            * 'goal_position_distance': myt.goal_position[0] - myt.position[0],
                            * 'goal_position_distance_absolute': abs(myt.goal_position[0] - myt.position[0])
                            * 'transmitted_comm': communication transmitted two the neighbours
                            * 'colour': colour of the top led
        """
        prox_values = myt.prox_values
        prox_comm = utils.get_prox_comm(myt)
        all_sensors = utils.get_all_sensors(prox_values, prox_comm)

        myt.dictionary = {
            'run': n_sim,
            'timestep': s,
            'name': myt.name,
            'index': myt.index,
            'myt_quantity': myt.controller.N,
            'net_input': myt.controller.net_input,
            'prox_values': prox_values,
            'prox_comm': utils.parse_prox_comm(prox_comm),
            'all_sensors': all_sensors,
            'initial_position': myt.initial_position,
            'position': myt.position,
            'angle': myt.angle,
            'goal_position': myt.goal_position,
            'goal_angle': myt.goal_angle,
            'motor_left_target': myt.motor_left_target,
            'motor_right_target': myt.motor_right_target,
            'goal_position_distance': myt.goal_position[0] - myt.position[0],
            'goal_position_distance_absolute': abs(myt.goal_position[0] - myt.position[0])
        }

        if comm:
            transmitted_comm = utils.get_transmitted_communication(myt)
            myt.dictionary['transmitted_comm'] = transmitted_comm

        if myt.controller.goal == 'colour':
            myt.dictionary['goal_colour'] = myt.goal_colour
            myt.dictionary['colour'] = myt.colour

        dictionary = myt.dictionary.copy()
        return dictionary

    @classmethod
    def update_dict(cls, myt, s, comm=None):
        """
        Updated data in the dictionary instead of rewrite every field to optimise performances.

        :param myt: agent
        :param s: timestep
        :param comm: boolean, True if the dataset is generated using the learned model with communication
        :return updated dictionary: updated dictionary
        """
        prox_values = myt.prox_values
        prox_comm = utils.get_prox_comm(myt)
        all_sensors = utils.get_all_sensors(prox_values, prox_comm)

        myt.dictionary.update(timestep=s),
        myt.dictionary['prox_values'] = prox_values
        myt.dictionary['prox_comm'] = utils.parse_prox_comm(prox_comm)
        myt.dictionary['all_sensors'] = all_sensors
        myt.dictionary['position'] = myt.position
        myt.dictionary['angle'] = myt.angle
        myt.dictionary['motor_left_target'] = myt.motor_left_target
        myt.dictionary['motor_right_target'] = myt.motor_right_target,
        myt.dictionary['goal_position_distance'] = myt.goal_position[0] - myt.position[0]
        myt.dictionary['goal_position_distance_absolute'] = abs(myt.goal_position[0] - myt.position[0])

        if comm:
            transmitted_comm = utils.get_transmitted_communication(myt)
            myt.dictionary['transmitted_comm'] = transmitted_comm

        if myt.controller.goal == 'colour':
            myt.dictionary['colour'] = myt.colour

        dictionary = myt.dictionary.copy()

        return dictionary

    @classmethod
    def save_simulation(cls, complete_data, data, runs_dir):
        """

        :param complete_data: list containing all the data of the robots in the simulations
        :param data: list containing all the data of the robots (except about the two robots at the beginning and
                     at the end of the row) in the simulations
        :param runs_dir: directory where to save the simulation runs
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
    def verify_target(cls, myt, counter, dictionary, tol):
        """
        Verify if the robot has reached the target goal.

        *It is always ensured that enough timesteps are performed, especially for the omniscient controller.*

        :param myt: agent
        :param counter: counter incremented when the robot has reached the goal
        :param dictionary: dictionary containing the agent information
        :param tol: tolerance
        """
        if myt.controller.goal == 'distribute' or myt.controller.name == 'omniscient':
            # do this also in case of omniscient controller to ensure enough timesteps
            diff = abs(dictionary['position'][0] - dictionary['goal_position'][0])
            if diff < tol:
                counter += 1

        elif myt.controller.goal == 'colour' and not myt.controller.name == 'omniscient':
            if myt.goal_colour == myt.colour:
                counter += 1

        else:
            raise ValueError("Invalid value for goal!")

        return counter

    @classmethod
    def run(cls, n_sim, myts, runs, complete_runs, world: pyenki.World,
            comm=None, gui: bool = False, T: float = 4, dt: float = 0.1, tol: float = 0.1) -> None:
        """
        Run the simulation as fast as possible or using the real time GUI.
        Generate two different type of simulation data, one with all the thymios and the other without including the 2
        thymios at the ends, but only the ones that have to move.
        If all the robots have reached their target, stop the simulation.

        :param n_sim: index of the simulation
        :param myts: list containing the agents
        :param runs: list containing all the data of the robots (except about the two robots at the beginning and
                     at the end of the row) in the simulations
        :param complete_runs: list containing all the data of the robots in the simulations
        :param world: enki world
        :param comm: variable that states if the communication is used
        :param gui: variable that states if use the gui
        :param T: duration of the simulation in seconds
        :param dt: control step duration – update timestep in seconds, should be below 1 (typically .02-.1)
        :param tol: tolerance used to verify if the robot reaches the target
        """

        myt_quantity = len(myts)

        if gui:
            # We can either run a simulation [in real-time] inside a Qt application
            world.run_in_viewer(cam_position=(70, 0), cam_altitude=170.0, cam_yaw=0.0, cam_pitch=-np.pi / 2,
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
                            dictionary = cls.generate_dict(myt, n_sim, s, comm)
                        else:
                            dictionary = cls.update_dict(myt, s, comm)

                        # Do not include the 2 thymio at the ends, but only the ones that have to move.
                        if i != 0 and i != (myt_quantity - 1):
                            iteration.append(dictionary)

                            # Check if the robot has reached the target, which depends on the goal:
                            # if it is distribute the thymios, then computed the euclidean distance between the actual
                            # and the goal position
                            # if it is colour the thymios, then check if the actual color is the same of the goal one
                            counter = cls.verify_target(myt, counter, dictionary, tol)

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
    def get_controller(cls, controller, controllers, goal, myt_quantity, net_input, communication=None, model=None, model_dir=None):
        """

        .. important:: This method has not yet been implemented for task 2.


        :param controller: name of the controller (between omniscient, manual and learned)
        :param controllers: reference to the controller class
        :param goal: task to perform (between colour ot distribute)
        :param myt_quantity: number of agents
        :param net_input: input of the net between prox_values, prox_comm or all_sensors
        :param communication: states if the communication is used by the network
        :param model: network to be used by the controller
        :param model_dir: directory containing the model
        :return controller_factory: lambda function containing the controller to be used
        """
        if controller == cls.LEARNED_CONTROLLER:
            net = torch.load('%s/%s' % (model_dir, model))
            # FIXME 'task2':
            controller_factory = lambda **kwargs: controllers.LearnedController(name=controller, goal=goal, N=myt_quantity,
                                                                                net=net, net_input=net_input,
                                                                                communication=communication, **kwargs)

        elif controller == cls.MANUAL_CONTROLLER:
            controller_factory = lambda **kwargs: controllers.ManualController(name=controller, goal=goal,
                                                                               N=myt_quantity, net_input=net_input,
                                                                               **kwargs)
        elif controller == cls.OMNISCIENT_CONTROLLER:
            controller_factory = lambda **kwargs: controllers.OmniscientController(name=controller, goal=goal,
                                                                                   N=myt_quantity, net_input=net_input,
                                                                                   **kwargs)
        else:
            raise ValueError("Invalid value for controller")

        return controller_factory

    @classmethod
    def generate_simulation(cls, run_dir, n_simulations, controller, myt_quantity, args, model_dir=None, model=None,
                            communication=False):
        """

        :param run_dir: directory containing the simulation runs
        :param n_simulations: number of simulations to generate
        :param controller: controller to be used in the simulations
        :param model_dir: directory containing the model
        :param myt_quantity: number of agents
        :param args: arguments from command line
        :param model: model to be used
        :param communication: states if the communication is used by the network
        """
        comm = False
        if communication:
            comm = True

        runs = []
        complete_runs = []

        if args.task == 'task1':
            from controllers import controllers_task1 as controllers
            goal = 'distribute'
            net_input = args.net_input
        else:
            from controllers import controllers_task2 as controllers
            goal = 'colour'
            net_input = None

        extension = True

        for n_sim in tqdm(range(n_simulations)):
            # if the number of agents is not defined randomly generate a simulation with N robots with N \in [5, 10]
            if args.myt_quantity == 'variable':
                myt_quantity = np.random.randint(5, 11)
            else:
                myt_quantity = int(args.myt_quantity)

            # if average gap is not defined randomly generate a simulation with avg_gap \in [5, 25]
            if args.avg_gap == 'variable':
                avg_gap = np.random.randint(5, 26)
            else:
                avg_gap = int(args.avg_gap)

            try:
                controller_factory = cls.get_controller(controller, controllers, goal, myt_quantity, args.net_input, communication, model, model_dir)

                world, myts = cls.setup(controller_factory, myt_quantity)
                cls.init_positions(myts, net_input, avg_gap, extension=extension)
                cls.run(n_sim, myts, runs, complete_runs, world, comm, args.gui)
            except Exception as e:
                print('ERROR: ', e)

        print('Saving dataset…')
        cls.save_simulation(complete_runs, runs, run_dir)

    @classmethod
    def check_dataset_conformity(cls, runs_dir, runs_img, title, dataset, net_input, communication):
        """
        Generate a scatter plot to check the conformity of the dataset.
        The plot will show the distribution of the input sensing, in particular, as the difference between the front
        sensor and the mean of the rear sensors, with respect to the output control of the datasets.

        :param runs_dir: directory containing the simulation
        :param runs_img: directory containing the simulation images
        :param title: title of the plot
        :param dataset: name of the dataset
        :param net_input: input of the net between prox_values, prox_comm or all_sensors
        :param communication: states if the communication is used by the network
        """

        runs = utils.load_dataset(runs_dir, 'simulation.pkl')
        runs_sub = runs[['timestep', 'run', 'motor_left_target', 'prox_values', 'prox_comm', 'all_sensors']]

        N = runs.myt_quantity.unique().max() - 2
        myt_quantitiy = np.array(runs[['run', 'myt_quantity']].drop_duplicates().myt_quantity) - 2

        x, y, myt_quantities, _, _ = utils.extract_input_output(runs_sub, net_input, N, myt_quantities=myt_quantitiy, communication=communication)

        #  Generate a scatter plot to check the conformity of the dataset
        file_name = 'dataset-scatterplot-%s' % dataset

        x_label = 'sensing (%s)' % net_input
        y_label = 'control'

        my_plots.my_scatterplot(x, y, x_label, y_label, runs_img, title, file_name)

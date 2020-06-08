import os
from itertools import chain
from operator import add

import numpy as np
import pandas as pd

import distributed
from distributed_thymio import DistributedThymio2


def check_dir(directory):
    """
    Check if the path is a directory, if not create it.
    :param directory: path to the directory
    """
    os.makedirs(directory, exist_ok=True)


def directory_for_dataset(dataset, controller):
    """

    :param dataset:
    :param controller:
    :return run_dir, run_img_dir, run_video_dir:
    """
    run_dir = os.path.join(dataset, controller)

    run_img_dir = os.path.join(run_dir, 'images')
    check_dir(run_img_dir)

    run_video_dir = os.path.join(run_dir, 'videos')
    check_dir(run_video_dir)

    return run_dir, run_img_dir, run_video_dir


def directory_for_model(args):
    """

    :param args:
    :return:
    """
    model_dir = os.path.join(args.models_folder, args.model_type, args.model)

    model_img_dir = os.path.join(model_dir, 'images')
    check_dir(model_img_dir)

    model_video_dir = os.path.join(model_dir, 'videos')
    check_dir(model_video_dir)

    metrics_path = os.path.join(model_dir, 'metrics.pkl')

    return model_dir, model_img_dir, model_video_dir, metrics_path


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


def signed_distance(state):
    """
    :return: Signed distance between current and the goal position, along the current theta of the robot
    """
    a = state.position[0] * np.cos(state.angle) + state.position[1] * np.sin(state.angle)
    b = state.goal_position[0] * np.cos(state.angle) + state.goal_position[1] * np.sin(state.angle)

    return b - a


def load_dataset(runs_dir, dataset):
    """

    :param runs_dir:
    :param dataset:
    :return dataframe:
    """
    pickle_file = os.path.join(runs_dir, dataset)
    runs = pd.read_pickle(pickle_file)

    flatten_runs = list(chain.from_iterable(list(chain.from_iterable(runs))))
    dataframe = pd.DataFrame(flatten_runs)

    return dataframe


def get_prox_comm(myt):
    """
    Create a dictionary containing all the senders as key and the corresponding intensities as value.
    :param myt
    :return prox_comm
    """
    prox_comm = {}

    prox_comm_events = myt.prox_comm_events

    if len(prox_comm_events) > 0:
        for idx, _ in enumerate(prox_comm_events):
            sender = prox_comm_events[idx].rx + 1
            intensities = prox_comm_events[idx].intensities

            prox_comm['myt%d' % sender] = {'intensities': intensities}

    return prox_comm


def parse_prox_comm(prox_comm):
    """

    :param prox_comm:
    :return prox_comm:
    """

    if len(prox_comm) == 0:
        prox_comm = [0, 0, 0, 0, 0, 0, 0]
    else:
        _, values = get_key_value_of_nested_dict(prox_comm)
        if len(prox_comm) == 1:
            prox_comm = values[0]
        else:
            prox_comm = np.max(np.array(values), axis=0).tolist()

    return prox_comm


def get_all_sensors(prox_values, prox_comm):
    """

    :param prox_values:
    :param prox_comm:
    :return all_sensors:
    """

    prox_comm = parse_prox_comm(prox_comm)

    all_sensors = list(map(add, prox_values, prox_comm))

    return all_sensors


def dataset_split(file_name, num_run=1000):
    """

    :param file_name:
    :param num_run:
    """
    x = np.arange(num_run)
    np.random.shuffle(x)

    np.save(file_name, x)


def get_input_sensing(in_label, myt, normalise=True):
    """

    :param in_label:
    :param myt:
    :param normalise
    :return sensing:
    """
    if isinstance(myt, dict):
        myt = distributed.ThymioState(myt)
    elif isinstance(myt, DistributedThymio2):
        if len(myt.prox_comm_events) == 0:
            prox_comm = {'sender': {'intensities': [0, 0, 0, 0, 0, 0, 0]}}
        else:
            prox_comm = get_prox_comm(myt)

        state_dict = {'initial_position': myt.initial_position, 'goal_position': myt.goal_position,
                      'prox_values': myt.prox_values, 'prox_comm': prox_comm}
        myt = distributed.ThymioState(state_dict)

    if in_label == 'prox_values':
        prox_values = getattr(myt, 'prox_values').copy()
        sensing = prox_values
    elif in_label == 'prox_comm':
        prox_comm = getattr(myt, 'prox_comm').copy()
        prox_comm = parse_prox_comm(prox_comm)

        sensing = prox_comm
    elif in_label == 'all_sensors':
        prox_values = getattr(myt, 'prox_values').copy()
        prox_comm = getattr(myt, 'prox_comm').copy()

        sensing = get_all_sensors(prox_values, prox_comm)
    else:
        raise ValueError("Invalid value for net_input")

    if normalise:
        sensing = np.divide(np.array(sensing), 1000).tolist()

    return sensing


def get_pos_sensing_control(runs_dir, net_input, distance_from_goal=None):
    """

    :param runs_dir:
    :param net_input
    :param distance_from_goal
    :return time_steps, x_positions, myt2_sensing, myt2_control, target
    """

    time_steps = []
    x_positions = []
    myt2_sensing = []
    myt2_control = []
    target = None
    mean_distances = None
    std_distances = None

    runs = load_dataset(runs_dir, 'complete-simulation.pkl')

    # FIXME

    for run in runs:
        run_time_steps = np.arange(len(run)).tolist()
        target = extract_run_data(myt2_control, myt2_sensing, run, time_steps, x_positions, net_input,
                                  distance_from_goal, run_time_steps)

    max_time_step = runs['timestep'].max()
    time_steps = np.arange(max_time_step)

    goal_p_dist_by_step = dataset_states.goal_position_distance.groupby('step')

    length2 = max(len(el) for el in list(chain(*x_positions)))
    for el1 in x_positions:
        el1.extend([[]] * (max_time_step - len(el1)))
        for el2 in el1:
            el2.extend([np.nan] * (length2 - len(el2)))
    x_positions = np.array(x_positions)

    length3 = max(len(el) for el in list(chain(*myt2_sensing)))
    for el1 in myt2_sensing:
        el1.extend([[]] * (max_time_step - len(el1)))
        for el2 in el1:
            el2.extend([np.nan] * (length3 - len(el2)))
    myt2_sensing = np.array(myt2_sensing)

    for el1 in myt2_control:
        el1.extend([np.nan] * (max_time_step - len(el1)))
    myt2_control = np.array(myt2_control)

    if distance_from_goal is not None:
        length4 = max(len(el) for el in list(chain(*distance_from_goal)))
        for el1 in distance_from_goal:
            el1.extend([[]] * (max_time_step - len(el1)))
            for el2 in el1:
                el2.extend([np.nan] * (length4 - len(el2)))
        distance_from_goal = np.array(np.abs(distance_from_goal))
        mean_distances = np.mean(distance_from_goal, axis=2)
        std_distances = np.std(distance_from_goal, axis=2)

    return time_steps, x_positions, myt2_sensing, myt2_control, target, mean_distances, std_distances


def get_key_value_of_nested_dict(nested_dict):
    """
    Access a nested dictionary and return a list of tuples (rv) and values. Used to return the list of intensities
    given a prox_comm dictionary containing multiple senders.
    :param nested_dict
    :return rv, values: rv is a list of tuples where, in each of these, the first element is a list of keys and the
    second is the final value. values is the list of inner values.
    """
    rv = []
    values = []
    for outer_key, value in nested_dict.items():
        try:
            inner_kvs, _ = get_key_value_of_nested_dict(value)
            for i_kvs in inner_kvs:
                rv.append((outer_key,) + i_kvs)
                values.append(i_kvs[1])
        except AttributeError:
            rv.append((outer_key, value))
            values.append(value)
    return rv, values


def extract_input_output(runs, in_label, input_combination=True):
    """
    Whether the input is prox_values, prox_comm or all sensors, it corresponds to the response values of ​​the
    sensors [array of 7 floats].
    The input is normalised so that the average is around 1 or a constant (e.g. for all (dividing by 1000)).
    The output is the speed of the wheels (which we assume equals left and right) [array of 1 float].
    There is no need to normalize the outputs.
    :param runs:
    :param in_label:
    :param input_combination
    :return input_, output_, runs
    """
    inputs = ['prox_values', 'prox_comm', 'all_sensors']
    inputs.remove(in_label)

    runs = runs.drop(columns=inputs)

    runs = pd.concat([runs.drop(in_label, axis=1),
                      pd.DataFrame(runs[in_label].to_list(), columns=['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br'])
                      # .add_prefix('%s_' % in_label)
                      ], axis=1)

    runs['x'] = runs.apply(lambda row: row.fc - np.mean([row.bl, row.br]), axis=1)
    runs[['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br', 'x']] = runs[['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br', 'x']].div(1000)
    # runs['x'] = runs.apply(lambda row: row.x / 1000, axis=1)

    if input_combination:
        input_ = np.array(runs.x)
    else:
        input_ = np.array(runs[['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']])
    output_ = np.array(runs.motor_left_target)

    return input_, output_, runs

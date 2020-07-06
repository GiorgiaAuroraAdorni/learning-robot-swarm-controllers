import os
from itertools import chain
from statistics import mean

import numpy as np
import pandas as pd

import train_distributed
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
    model_dir = os.path.join(args.models_folder, args.task, args.model_type, args.model)

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

    all_sensors = prox_values + prox_comm

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
        myt = train_distributed.ThymioState(myt)
    elif isinstance(myt, DistributedThymio2):
        if len(myt.prox_comm_events) == 0:
            prox_comm = {'sender': {'intensities': [0, 0, 0, 0, 0, 0, 0]}}
        else:
            prox_comm = get_prox_comm(myt)

        state_dict = {'initial_position': myt.initial_position, 'goal_position': myt.goal_position,
                      'prox_values': myt.prox_values, 'prox_comm': prox_comm}
        myt = train_distributed.ThymioState(state_dict)

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


def get_input_columns(in_label):
    """

    :param in_label:
    :return columns:
    """
    if in_label == 'all_sensors':
        columns = ['pv_fll', 'pv_fl', 'pv_fc', 'pv_fr', 'pv_frr', 'pv_bl', 'pv_br',
                   'pc_fll', 'pc_fl', 'pc_fc', 'pc_fr', 'pc_frr', 'pc_bl', 'pc_br']
    else:
        columns = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']\

    return columns


def extract_input_output(runs, in_label, input_combination=True, communication=False):
    """
    Whether the input is prox_values, prox_comm or all sensors, it corresponds to the response values of ​​the
    sensors [array of 7 floats].
    The input is normalised so that the average is around 1 or a constant (e.g. for all (dividing by 1000)).
    The output is the speed of the wheels (which we assume equals left and right) [array of 1 float].
    There is no need to normalize the outputs.
    :param runs:
    :param in_label:
    :param input_combination
    :param communication
    :return input_, output_, runs, columns
    """
    inputs = ['prox_values', 'prox_comm', 'all_sensors']
    inputs.remove(in_label)

    runs = runs.drop(columns=inputs)

    columns = get_input_columns(in_label)

    runs = pd.concat([runs.drop(in_label, axis=1),
                      pd.DataFrame(runs[in_label].to_list(), columns=columns)
                      # .add_prefix('%s_' % in_label)
                      ], axis=1)

    full_columns = columns + ['x']

    if input_combination:
        if in_label == 'all_sensors':
            runs['x'] = runs.apply(lambda row: mean([row.pv_fc - mean([row.pv_bl, row.pv_br]), row.pc_fc - mean([row.pc_bl, row.pc_br])]), axis=1)
        else:
            runs['x'] = runs.apply(lambda row: row.fc - mean([row.bl, row.br]), axis=1)

        runs[full_columns] = runs[full_columns].div(1000)

        input_ = np.array(runs.x)
        output_ = np.array(runs.motor_left_target)
    else:
        runs[columns] = runs[columns].div(1000)

        if communication:
            simulations = runs['run'].unique()

            tmp = np.array(runs[['run', 'timestep']].drop_duplicates().groupby(['run']).max()).squeeze()
            timesteps = np.sum(tmp) - tmp.shape[0]

            input_ = np.empty(shape=(timesteps, 2, 3, 7), dtype='float32')
            output_ = np.empty(shape=(timesteps, 2, 3, 1), dtype='float32')

            init_counter = 0
            for i in simulations:
                run = runs[runs['run'] == i]

                in_run_ = np.array(run[columns])
                in_run_ = in_run_.reshape([-1, 3, 7])
                out_run_ = np.array(run.motor_left_target)
                out_run_ = out_run_.reshape([-1, 3, 1])

                size = in_run_.shape[0] - 1
                final_counter = init_counter + size

                in_array = np.empty(shape=(size, 2, 3, 7), dtype='float32')
                in_array[:, 0, ...] = in_run_[:-1, ...]
                in_array[:, 1, ...] = in_run_[1:, ...]

                out_array = np.empty(shape=(size, 2, 3, 1), dtype='float32')
                out_array[:, 0, ...] = out_run_[:-1, ...]
                out_array[:, 1, ...] = out_run_[1:, ...]

                input_[init_counter:final_counter] = in_array
                output_[init_counter:final_counter] = out_array

                init_counter = final_counter
        else:
            input_ = np.array(runs[columns])
            output_ = np.array(runs.motor_left_target)

    return input_, output_, runs, columns

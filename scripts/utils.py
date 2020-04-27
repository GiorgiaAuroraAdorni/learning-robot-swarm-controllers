import os
from itertools import chain

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


def signed_distance(state):
    """
    :return: Signed distance between current and the goal position, along the current theta of the robot
    """
    a = state.position[0] * np.cos(state.angle) + state.position[1] * np.sin(state.angle)
    b = state.goal_position[0] * np.cos(state.angle) + state.goal_position[1] * np.sin(state.angle)

    return b - a


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
    :return:
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

    prox_values = getattr(myt, 'prox_values').copy()
    prox_comm = getattr(myt, 'prox_comm').copy()

    if len(prox_comm) == 0:
        prox_comm = [0, 0, 0, 0, 0, 0, 0]
    else:
        _, values = get_key_value_of_nested_dict(prox_comm)
        if len(prox_comm) == 1:
            prox_comm = values[0]
        else:
            prox_comm = np.max(np.array(values), axis=0).tolist()

    if normalise:
        prox_values = np.divide(np.array(prox_values), 1000).tolist()
        prox_comm = np.divide(np.array(prox_comm), 1000).tolist()

    if in_label == 'prox_values':
        sensing = prox_values
    elif in_label == 'prox_comm':
        sensing = prox_comm
    elif in_label == 'all_sensors':
        sensing = prox_values + prox_comm
    else:
        raise ValueError("Invalid value for net_input")

    return sensing


def extract_run_data(myt2_control, myt2_sensing, run, time_steps, x_positions, net_input, distance_from_goal=None,
                     run_time_steps=None):
    """

    :param myt2_control:
    :param myt2_sensing:
    :param run:
    :param time_steps:
    :param x_positions:
    :param net_input
    :param distance_from_goal
    :param run_time_steps:
    """
    run_x_positions = []
    run_myt2_sensing = []
    run_myt2_control = []

    target = []

    counter = 0
    for step in run:
        x_pos = []
        for myt in step:
            x_position = myt['position'][0]
            x_pos.append(x_position)

            # Append goal_position just one time
            if counter == 0:
                x_t = myt['goal_position'][0]
                target.append(x_t)

            if myt['name'] == 'myt2':
                sensing = get_input_sensing(net_input, myt)
                control = myt['motor_left_target']

                run_myt2_sensing.append(sensing)
                run_myt2_control.append(control)

        counter += 1
        run_x_positions.append(x_pos)

    if run_time_steps is not None:
        time_steps.append(run_time_steps)
    x_positions.append(run_x_positions)
    myt2_sensing.append(run_myt2_sensing)
    myt2_control.append(run_myt2_control)

    if distance_from_goal is not None:
        distance_from_goal.append((np.array(run_x_positions) - np.array(target)).tolist())

    return target


def get_pos_sensing_control(runs_dir, net_input, distance_from_goal=None):
    """

    :param runs_dir:
    :param distance_from_goal
    :return time_steps, x_positions, myt2_sensing, myt2_control, target
    """

    time_steps = []
    x_positions = []
    myt2_sensing = []
    myt2_control = []
    target = None
    mean_distances = None

    pickle_file = os.path.join(runs_dir, 'complete-simulation.pkl')
    runs = pd.read_pickle(pickle_file)

    for run in runs:
        run_time_steps = np.arange(len(run)).tolist()
        target = extract_run_data(myt2_control, myt2_sensing, run, time_steps, x_positions, net_input,
                                  distance_from_goal, run_time_steps)

    length = max(map(len, time_steps))
    time_steps = np.arange(length)

    length2 = max(len(el) for el in list(chain(*x_positions)))
    for el1 in x_positions:
        el1.extend([[]] * (length - len(el1)))
        for el2 in el1:
            el2.extend([np.nan] * (length2 - len(el2)))
    x_positions = np.array(x_positions)

    length3 = max(len(el) for el in list(chain(*myt2_sensing)))
    for el1 in myt2_sensing:
        el1.extend([[]] * (length - len(el1)))
        for el2 in el1:
            el2.extend([np.nan] * (length3 - len(el2)))
    myt2_sensing = np.array(myt2_sensing)

    for el1 in myt2_control:
        el1.extend([np.nan] * (length - len(el1)))
    myt2_control = np.array(myt2_control)

    if distance_from_goal is not None:
        length4 = max(len(el) for el in list(chain(*distance_from_goal)))
        for el1 in distance_from_goal:
            el1.extend([[]] * (length - len(el1)))
            for el2 in el1:
                el2.extend([np.nan] * (length4 - len(el2)))
        distance_from_goal = np.array(np.abs(distance_from_goal))
        mean_distances = np.mean(distance_from_goal, axis=2)

    return time_steps, x_positions, myt2_sensing, myt2_control, target, mean_distances


def extract_flatten_dataframe(myt2_control, myt2_sensing, time_steps, x_positions):
    """

    :param myt2_control:
    :param myt2_sensing:
    :param time_steps:
    :param x_positions:
    :return df_x_positions, df_sensing, df_control
    """
    df_control = {'timestep': np.array([time_steps] * 1000).flatten(),
                  'myt2_control': myt2_control.flatten()}
    df_control = pd.DataFrame(data=df_control)

    flat_x_positions = x_positions.reshape(-1, x_positions.shape[-1])

    df_x_positions = {'timestep': np.array([time_steps] * 1000).flatten()}
    for i in range(flat_x_positions.shape[1]):
        df_x_positions['myt%d_x_positions' % (i + 2)] = flat_x_positions[:, i]
    df_x_positions = pd.DataFrame(data=df_x_positions)

    flat_myt2_sensing = myt2_sensing.reshape(-1, myt2_sensing.shape[-1])
    df_sensing = {'timestep': np.array([time_steps] * 1000).flatten()}
    for i in range(flat_myt2_sensing.shape[1]):
        df_sensing['myt2_sensor_%d' % (i + 1)] = flat_myt2_sensing[:, i]
    df_sensing = pd.DataFrame(data=df_sensing)

    return df_x_positions, df_sensing, df_control


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


def extract_input_output(run, input_, output_, in_label, out_label):
    """
    The input is the prox_values, that are the response values of ​​the sensors [array of 7 floats],
    they are normalised so that the average is around 1 or a constant (e.g. for all (dividing by 1000)).
    The output is the speed of the wheels (which we assume equals left and right) [array of 1 float].
    There is no need to normalize the outputs.
    :param run:
    :param input_:
    :param output_:
    :param in_label:
    :param out_label:
    """
    for step in run:
        for myt in step:
            sensing = get_input_sensing(in_label, myt)
            input_.append(sensing)

            speed = myt[out_label]
            output_.append([speed])


def extract_input(run, sensing, net_input):
    """

    :param run:
    :param sensing:
    :param net_input:
    """
    run_sensing = []

    for step in run:
        step_sensing = []
        for myt in step:
            s = get_input_sensing(net_input, myt)
            step_sensing.append(s)
        run_sensing.append(step_sensing)

    sensing.append(run_sensing)


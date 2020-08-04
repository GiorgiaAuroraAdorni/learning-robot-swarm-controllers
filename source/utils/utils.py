import os
from itertools import chain
from statistics import mean

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


class ThymioState:
    def __init__(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)


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

    prox_comm_events = myt.prox_comm_events.copy()

    if len(prox_comm_events) > 0:
        for idx, _ in enumerate(prox_comm_events):
            intensities = prox_comm_events[idx].intensities

            if mean([intensities[5], intensities[6]]) != 0:
                sender = myt.index
            else:
                sender = myt.index + 2

            prox_comm['myt%d' % sender] = {'intensities': intensities}

    return prox_comm


def get_received_communication(myt, goal='distribute'):
    """
    Create a list containing the messages received from the back and front.
    :param myt
    :param goal
    :return communication: the communication received
    """
    communication = [0, 0]

    prox_comm_events = myt.prox_comm_events.copy()

    for idx, _ in enumerate(prox_comm_events):
        message = prox_comm_events[idx].rx
        if goal == 'distribute':
            message = float(message / (2 ** 10))

        if mean([prox_comm_events[idx].intensities[5], prox_comm_events[idx].intensities[6]]) != 0:
            communication[0] = message
        else:
            communication[1] = message

    return communication.copy()


def get_transmitted_communication(myt):
    """
    Return the values transmitted during the communication.
    :param myt
    :return communication: the communication transmitted
    """
    communication = myt.prox_comm_tx
    return communication


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
    from thymio import DistributedThymio2

    if isinstance(myt, dict):
        myt = ThymioState(myt)
    elif isinstance(myt, DistributedThymio2):
        if len(myt.prox_comm_events) == 0:
            prox_comm = {'sender': {'intensities': [0, 0, 0, 0, 0, 0, 0]}}
        else:
            prox_comm = get_prox_comm(myt)

        state_dict = {'initial_position': myt.initial_position, 'goal_position': myt.goal_position,
                      'prox_values': myt.prox_values, 'prox_comm': prox_comm}
        myt = ThymioState(state_dict)

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


def prepare_dataset(run_dir, split):
    """
    :param run_dir
    :param split
    :return file, indices
    """
    file = os.path.join(run_dir, 'dataset_split.npy')
    # Uncomment the following line to generate a new dataset split

    if split:
        dataset_split(file)

    # Load the indices
    dataset = np.load(file)
    n_train = 600
    n_validation = 800
    train_indices, validation_indices, test_indices = dataset[:n_train], dataset[n_train:n_validation], \
                                                      dataset[n_validation:]

    indices = [train_indices, validation_indices, test_indices]

    return file, indices


def from_indices_to_dataset(runs_dir, train_indices, validation_indices, test_indices, net_input, communication=False):
    """
    :param runs_dir: directory containing the simulations
    :param train_indices
    :param validation_indices
    :param test_indices
    :param net_input
    :param communication
    :return: (train_sample, valid_sample, test_sample), train_target, valid_target, test_target, train_quantities, valid_quantities, test_quantities
    """
    runs = load_dataset(runs_dir, 'simulation.pkl')

    N = runs.myt_quantity.unique().max() - 2
    myt_quantities = np.array(runs[['run', 'myt_quantity']].drop_duplicates().myt_quantity) - 2

    # For old datasets
    # N = 3
    # myt_quantities = np.full(shape=(1000,), fill_value=N, dtype='float32')

    if communication:
        runs_sub = runs[['timestep', 'name', 'run', 'motor_left_target', 'prox_values', 'prox_comm', 'all_sensors']]
    else:
        runs_sub = runs[['timestep', 'myt_quantity', 'run', 'motor_left_target', 'prox_values', 'prox_comm', 'all_sensors']]

    train_runs = runs_sub[runs_sub['run'].isin(train_indices)].reset_index()
    valid_runs = runs_sub[runs_sub['run'].isin(validation_indices)].reset_index()
    test_runs = runs_sub[runs_sub['run'].isin(test_indices)].reset_index()

    train_sample, train_target, train_quantities, _, _ = extract_input_output(train_runs, net_input, N=N,
                                                                              communication=communication,
                                                                              input_combination=False,
                                                                              myt_quantities=myt_quantities)
    valid_sample, valid_target, valid_quantities, _, _ = extract_input_output(valid_runs, net_input, N=N,
                                                                              communication=communication,
                                                                              input_combination=False,
                                                                              myt_quantities=myt_quantities)
    test_sample, test_target, test_quantities, _, _ = extract_input_output(test_runs, net_input, N=N,
                                                                           communication=communication,
                                                                           input_combination=False,
                                                                           myt_quantities=myt_quantities)

    return train_sample, valid_sample, test_sample, train_target, valid_target, test_target, train_quantities, valid_quantities, test_quantities


def from_dataset_to_tensors(train_sample, train_target, valid_sample, valid_target, test_sample, test_target, q_train, q_valid, q_test):
    """

    :param train_sample:
    :param train_target:
    :param valid_sample:
    :param valid_target:
    :param test_sample:
    :param test_target:
    :param q_train:
    :param q_valid:
    :param q_test:
    :return test, train, valid
    """
    x_train_tensor = torch.tensor(train_sample, dtype=torch.float32)
    x_valid_tensor = torch.tensor(valid_sample, dtype=torch.float32)
    x_test_tensor = torch.tensor(test_sample, dtype=torch.float32)

    y_train_tensor = torch.tensor(train_target, dtype=torch.float32)
    y_valid_tensor = torch.tensor(valid_target, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_target, dtype=torch.float32)

    q_train_tensor = torch.tensor(q_train, dtype=torch.float32)
    q_valid_tensor = torch.tensor(q_valid, dtype=torch.float32)
    q_test_tensor = torch.tensor(q_test, dtype=torch.float32)

    train = TensorDataset(x_train_tensor, y_train_tensor, q_train_tensor)
    valid = TensorDataset(x_valid_tensor, y_valid_tensor, q_valid_tensor)
    test = TensorDataset(x_test_tensor, y_test_tensor, q_test_tensor)

    return test, train, valid


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


def extract_input_output(runs, in_label, N, communication=False, input_combination=True, myt_quantities=None):
    """
    Whether the input is prox_values, prox_comm or all sensors, it corresponds to the response values of ​​the
    sensors [array of 7 floats].
    The input is normalised so that the average is around 1 or a constant (e.g. for all (dividing by 1000)).
    The output is the speed of the wheels (which we assume equals left and right) [array of 1 float].
    There is no need to normalize the outputs.
    :param runs:
    :param in_label:
    :param N
    :param communication
    :param input_combination
    :param myt_quantities
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
        myt_quantities_ = None
    else:
        runs[columns] = runs[columns].div(1000)

        if communication:
            simulations = runs['run'].unique()

            tmp = np.array(runs[['run', 'timestep']].drop_duplicates().groupby(['run']).max()).squeeze()
            timesteps = np.sum(tmp) - tmp.shape[0]

            input_ = np.empty(shape=(timesteps, 2, N, runs[columns].shape[1]), dtype='float32')
            output_ = np.empty(shape=(timesteps, 2, N), dtype='float32')
            myt_quantities_ = np.empty(shape=(timesteps, 2, N), dtype='float32')

            init_counter = 0
            for i in simulations:
                N_sim = int(myt_quantities[i])
                run = runs[runs['run'] == i]

                in_run_ = np.array(run[columns])
                in_run_ = in_run_.reshape([-1, N_sim, run[columns].shape[1]])
                out_run_ = np.array(run.motor_left_target)
                out_run_ = out_run_.reshape([-1, N_sim])

                size = in_run_.shape[0] - 1
                final_counter = init_counter + size

                in_array = np.empty(shape=(size, 2, N_sim, run[columns].shape[1]), dtype='float32')
                in_array[:, 0, ...] = in_run_[:-1, ...]
                in_array[:, 1, ...] = in_run_[1:, ...]

                out_array = np.empty(shape=(size, 2, N_sim), dtype='float32')
                out_array[:, 0, ...] = out_run_[:-1, ...]
                out_array[:, 1, ...] = out_run_[1:, ...]

                # Padded array for reach the maximum number of thymios (for task1 extension)
                in_array = np.pad(in_array, ((0, 0), (0, 0), (0, N - N_sim), (0, 0)), 'constant', constant_values=np.nan)
                out_array = np.pad(out_array, ((0, 0), (0, 0), (0, N - N_sim)), 'constant', constant_values=np.nan)
                myt = np.full(shape=(size, 2, N), fill_value=N_sim, dtype='float32')

                input_[init_counter:final_counter] = in_array
                output_[init_counter:final_counter] = out_array
                myt_quantities_[init_counter:final_counter] = myt

                init_counter = final_counter
        else:
            input_ = np.array(runs[columns])
            output_ = np.array(runs.motor_left_target)
            myt_quantities_ = np.array(runs.myt_quantity)

    return input_, output_, myt_quantities_, runs, columns


def extract_colour_output(runs, communication=False, input_combination=True):
    """
    The output is the colour of the top led that depends by the position of the robot in the row.
    :param runs:
    :param communication
    :param input_combination
    :return output_, runs
    """
    if input_combination:
        output_ = np.array(runs.goal_colour)
    else:
        if communication:
            simulations = runs['run'].unique()

            tmp = np.array(runs[['run', 'timestep']].drop_duplicates().groupby(['run']).max()).squeeze()
            timesteps = np.sum(tmp) - tmp.shape[0]

            output_ = np.empty(shape=(timesteps, 2, 3), dtype='float32')

            init_counter = 0
            for i in simulations:
                run = runs[runs['run'] == i]

                out_run_ = np.array(run.goal_colour)
                out_run_ = out_run_.reshape([-1, 3])

                size = out_run_.shape[0] - 1
                final_counter = init_counter + size

                out_array = np.empty(shape=(size, 2, 3), dtype='float32')
                out_array[:, 0, ...] = out_run_[:-1, ...]
                out_array[:, 1, ...] = out_run_[1:, ...]

                output_[init_counter:final_counter] = out_array

                init_counter = final_counter
        else:
            output_ = np.array(runs.motor_left_target)

    return output_, runs


def export_network(model_dir, model, input_):
    """

    :param model_dir:
    :param model:
    :param input_:
    :return:
    """
    net = torch.load('%s/%s' % (model_dir, model), map_location='cpu')

    # torch.save(net.single_net, '%s/%s.single-net' % (model_dir, model))

    # Export the model
    torch.onnx.export(net,                                        # model being run
                      input_,                                     # model input (or a tuple for multiple inputs)
                      "%s/%s.onnx" % (model_dir, model),           # where to save the model (can be a file or file-like object)
                      do_constant_folding=True,
                      # export_params=True,                         # store the trained parameter weights inside the model file
                      # opset_version=10                           # the ONNX version to export the model to
                      input_names=['sensing'],  # the model's input names
                      output_names=['control']  # the model's output names
                      )


def write_graph(model_dir, model, dummy_input_graph):
    """
    FIXME
    :param model_dir:
    :param model:
    :param input_:
    :return:
    """
    from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('%s/tb' % model_dir)

    net = torch.load('%s/%s' % (model_dir, model))

    writer.add_graph(net, dummy_input_graph)
    writer.close()


def generate_fake_simulations(run_dir, model, initial_positions, myt_quantity):
    """
    """

    from controllers import controllers_task1 as controllers
    from generate_simulation_data import GenerateSimulationData as g

    goal_positions = np.linspace(0, initial_positions[-1], num=myt_quantity, dtype=np.float64)

    goal = 'distribute'
    net_input = 'all_sensors'

    omniscient_controller_factory = g.get_controller('omniscient', controllers, goal, myt_quantity, net_input)

    manual_controller_factory = g.get_controller('manual', controllers, goal, myt_quantity, net_input)

    distributed_net_dir = os.path.join('models', 'task1', 'distributed', model)
    distributed_controller_factory = g.get_controller('learned', controllers, goal, myt_quantity, net_input,
                                                      model=model, model_dir=distributed_net_dir,
                                                      communication=False)

    communication_net_dir = os.path.join('models', 'task1', 'communication', model)
    communication_controller_factory = g.get_controller('learned', controllers, goal, myt_quantity, net_input,
                                                        model=model, model_dir=communication_net_dir,
                                                        communication=True)

    controller_factories = [(omniscient_controller_factory, 'omniscient'),
                            (manual_controller_factory, 'manual'),
                            (distributed_controller_factory, 'distributed'),
                            (communication_controller_factory, 'communication')]

    out_dirs = []
    for factory, name in controller_factories:
        world, myts = g.setup(factory, myt_quantity)

        comm = False
        if myts[0].controller.name == 'learned':
            if myts[0].controller.communication:
                comm = True

        for i, myt in enumerate(myts):
            # Position the first and last robot at a fixed distance
            myt.position = (initial_positions[i], 0)
            myt.initial_position = myt.position

            # Reset the parameters
            myt.dictionary = None
            myt.angle = 0
            myt.goal_position = (goal_positions[i], 0)

            if myt.colour is not None:
                myt.colour = None
            myt.prox_comm_tx = 0
            myt.prox_comm_enable = False

        runs = []
        complete_runs = []

        g.run(0, myts, runs, complete_runs, world, comm, gui=False)

        out_dir = os.path.join(run_dir, name)
        check_dir(out_dir)
        g.save_simulation(complete_runs, runs, out_dir)

        out_dirs.append(out_dir)

    return out_dirs

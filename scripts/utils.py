import os
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="white")


def check_dir(directory):
    """
    Check if the path is a directory, if not create it.
    :param directory: path to the directory
    """
    os.makedirs(directory, exist_ok=True)


def dataset_split(file_name, num_run=1000):
    """

    :param file_name:
    :param num_run:
    """
    x = np.arange(num_run)
    np.random.shuffle(x)

    np.save(file_name, x)


def visualise_simulation(runs_dir, img_dir, model):
    """
    :param runs_dir:
    :param img_dir:
    :param model:
    """
    runs_dir = os.path.join(runs_dir, model)
    run = None

    x_positions = []
    myt2_sensing = []
    myt2_control = []

    for file_name in os.listdir(runs_dir):
        if not file_name.endswith('-0.pkl'):
            continue

        pickle_file = os.path.join(runs_dir, file_name)
        run = pd.read_pickle(pickle_file)

    time_steps = np.arange(len(run))
    extract_run_data(myt2_control, myt2_sensing, run, time_steps, x_positions)

    x_positions = np.array(x_positions[0])
    myt2_sensing = np.array(myt2_sensing[0])
    myt2_control = np.array(myt2_control[0])

    grid = np.linspace(0, x_positions[-1, -1], 5)
    fig, axes = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)

    # Plot the evolution of the positions of all robots over time
    # axes[0].set_xlabel('timestep', fontsize=11)
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    for i in range(np.shape(x_positions)[1]):
        axes[0].plot(time_steps, x_positions[:, i], label='myt%d' % (i + 1))  # , color='black')
    axes[0].legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    axes[0].set_yticks(grid)
    axes[0].grid()

    # Plot, for a given robot, the evolution of sensing over time
    # axes[1].set_xlabel('timestep', fontsize=11)
    axes[1].set_ylabel('sensing', fontsize=11)
    axes[1].set_title('Thymio 2 Sensing', weight='bold', fontsize=12)
    for i in range(np.shape(myt2_sensing)[1]):
        axes[1].plot(time_steps, myt2_sensing[:, i], label='prox sensor %d' % (i + 1))  # , color='black')
    axes[1].legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    axes[1].grid()

    # Plot, for a given robot, the evolution of control over time
    # axes[2].set_xlabel('timestep', fontsize=11)
    axes[2].set_ylabel('control', fontsize=11)
    axes[2].set_title('Thymio 2 Control', weight='bold', fontsize=12)
    axes[2].plot(time_steps, myt2_control, color='black')
    axes[2].grid()

    plt.tight_layout()
    plt.xlabel('timestep', fontsize=11)

    check_dir(img_dir)
    filename = 'plot-simulation-%s.png' % model
    file = os.path.join(img_dir, filename)
    plt.savefig(file)
    plt.show()


def extract_run_data(myt2_control, myt2_sensing, run, time_steps, x_positions, run_time_steps=None):
    """

    :param myt2_control:
    :param myt2_sensing:
    :param run:
    :param time_steps:
    :param x_positions:
    :param run_time_steps:
    """
    run_x_positions = []
    run_myt2_sensing = []
    run_myt2_control = []

    for step in run:
        x_pos = []

        for myt in step:
            x_position = myt['position'][0]
            x_pos.append(x_position)

            if myt['name'] == 'myt2':
                sensing = myt['prox_values'].copy()
                control = myt['motor_left_target']

                run_myt2_sensing.append(sensing)
                run_myt2_control.append(control)

        run_x_positions.append(x_pos)

    if run_time_steps is not None:
        time_steps.append(run_time_steps)
    x_positions.append(run_x_positions)
    myt2_sensing.append(run_myt2_sensing)
    myt2_control.append(run_myt2_control)


def get_pos_sensing_control(model, runs_dir):
    """

    :param model:
    :param runs_dir:
    :return:
    """
    runs_dir = os.path.join(runs_dir, model)

    time_steps = []
    x_positions = []
    myt2_sensing = []
    myt2_control = []

    for file_name in os.listdir(runs_dir):
        if not file_name.endswith('.pkl'):
            continue

        pickle_file = os.path.join(runs_dir, file_name)
        run = pd.read_pickle(pickle_file)

        run_time_steps = np.arange(len(run)).tolist()
        extract_run_data(myt2_control, myt2_sensing, run, time_steps, x_positions, run_time_steps)

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

    return time_steps, x_positions, myt2_sensing, myt2_control


def extract_flatten_dataframe(myt2_control, myt2_sensing, time_steps, x_positions):
    """

    :param myt2_control:
    :param myt2_sensing:
    :param time_steps:
    :param x_positions:
    :return:
    """
    df_control = {'timestep': np.array([time_steps] * 1000).flatten(),
                  'myt2_control': myt2_control.flatten()}
    df_control = pd.DataFrame(data=df_control)

    flat_x_positions = x_positions.reshape(-1, x_positions.shape[-1])
    df_x_positions = {'timestep': np.array([time_steps] * 1000).flatten(),
                      'myt1_x_positions': flat_x_positions[:, 0],
                      'myt2_x_positions': flat_x_positions[:, 1],
                      'myt3_x_positions': flat_x_positions[:, 2],
                      'myt4_x_positions': flat_x_positions[:, 3],
                      'myt5_x_positions': flat_x_positions[:, 4]}
    df_x_positions = pd.DataFrame(data=df_x_positions)

    flat_myt2_sensing = myt2_sensing.reshape(-1, myt2_sensing.shape[-1])
    df_sensing = {'timestep': np.array([time_steps] * 1000).flatten(),
                  'myt2_sensor_1': flat_myt2_sensing[:, 0],
                  'myt2_sensor_2': flat_myt2_sensing[:, 1],
                  'myt2_sensor_3': flat_myt2_sensing[:, 2],
                  'myt2_sensor_4': flat_myt2_sensing[:, 3],
                  'myt2_sensor_5': flat_myt2_sensing[:, 4],
                  'myt2_sensor_6': flat_myt2_sensing[:, 5],
                  'myt2_sensor_7': flat_myt2_sensing[:, 6]}
    df_sensing = pd.DataFrame(data=df_sensing)

    return df_x_positions, df_sensing, df_control


def visualise_simulations_comparison(runs_dir, img_dir, model):
    """
    :param runs_dir:
    :param img_dir:
    :param model:
    """
    time_steps, x_positions, myt2_sensing, myt2_control = get_pos_sensing_control(model, runs_dir)

    mean_x_positions = np.nanmean(x_positions, axis=0)
    mean_myt2_control = np.nanmean(myt2_control, axis=0)
    mean_myt2_sensing = np.nanmean(myt2_sensing, axis=0)

    std_x_positions = np.nanstd(x_positions, axis=0)
    std_myt2_control = np.nanstd(myt2_control, axis=0)
    std_myt2_sensing = np.nanstd(myt2_sensing, axis=0)

    grid = np.linspace(0, mean_x_positions[-1, -1], 5)
    fig, axes = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)

    # Plot the evolution of the positions of all robots over time
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    for i in range(np.shape(mean_x_positions)[1]):
        axes[0].plot(time_steps, mean_x_positions[:, i], label='myt%d' % (i + 1))
        axes[0].fill_between(time_steps,
                             mean_x_positions[:, i] - std_x_positions[:, i],
                             mean_x_positions[:, i] + std_x_positions[:, i],
                             alpha=0.2)
    axes[0].legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    axes[0].set_yticks(grid)
    axes[0].grid()

    # Plot, for a given robot, the evolution of sensing over time
    axes[1].set_ylabel('sensing', fontsize=11)
    axes[1].set_title('Thymio 2 Sensing', weight='bold', fontsize=12)
    for i in range(np.shape(mean_myt2_sensing)[1]):
        axes[1].plot(time_steps, mean_myt2_sensing[:, i], label='prox sensor %d' % (i + 1))  # , color='black')
        axes[1].fill_between(time_steps,
                             mean_myt2_sensing[:, i] - std_myt2_sensing[:, i],
                             mean_myt2_sensing[:, i] + std_myt2_sensing[:, i],
                             alpha=0.2)
    axes[1].legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
    axes[1].grid()

    # Plot, for a given robot, the evolution of control over time
    axes[2].set_ylabel('control', fontsize=11)
    axes[2].set_title('Thymio 2 Control', weight='bold', fontsize=12)
    axes[2].plot(time_steps, mean_myt2_control, color='black')
    axes[2].fill_between(time_steps,
                         mean_myt2_control - std_myt2_control,
                         mean_myt2_control + std_myt2_control,
                         alpha=0.2)
    axes[2].grid()

    plt.tight_layout()
    plt.xlabel('timestep', fontsize=11)

    check_dir(img_dir)
    filename = 'compare-simulation-%s.png' % model
    file = os.path.join(img_dir, filename)
    plt.savefig(file)
    plt.show()

    # Seaborn visualisation
    df_x_positions, df_sensing, df_control = extract_flatten_dataframe(myt2_control, myt2_sensing, time_steps,
                                                                       x_positions)

    fig, axes = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)
    sns.lineplot(x="timestep", y="myt1_x_positions", data=df_x_positions, ax=axes[0])
    sns.lineplot(x="timestep", y="myt2_x_positions", data=df_x_positions, ax=axes[0])
    sns.lineplot(x="timestep", y="myt3_x_positions", data=df_x_positions, ax=axes[0])
    sns.lineplot(x="timestep", y="myt4_x_positions", data=df_x_positions, ax=axes[0])
    sns.lineplot(x="timestep", y="myt5_x_positions", data=df_x_positions, ax=axes[0])
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    axes[0].legend(loc='center right', bbox_to_anchor=(1.3, 0.5),
                   labels=('myt1', 'myt2', 'myt3', 'myt4', 'myt5'))
    axes[0].set_yticks(grid)
    axes[0].grid()
    axes[0].set(ylabel='x position')

    sns.lineplot(x="timestep", y="myt2_sensor_1", data=df_sensing, ax=axes[1])
    sns.lineplot(x="timestep", y="myt2_sensor_2", data=df_sensing, ax=axes[1])
    sns.lineplot(x="timestep", y="myt2_sensor_3", data=df_sensing, ax=axes[1])
    sns.lineplot(x="timestep", y="myt2_sensor_4", data=df_sensing, ax=axes[1])
    sns.lineplot(x="timestep", y="myt2_sensor_5", data=df_sensing, ax=axes[1])
    sns.lineplot(x="timestep", y="myt2_sensor_6", data=df_sensing, ax=axes[1])
    sns.lineplot(x="timestep", y="myt2_sensor_7", data=df_sensing, ax=axes[1])
    axes[1].set_title('Thymio 2 Sensing', weight='bold', fontsize=12)
    axes[1].legend(loc='center right', bbox_to_anchor=(1.35, 0.5),
                   labels=('prox sensor 1', 'prox sensor 2', 'prox sensor 3', 'prox sensor 4', 'prox sensor 5',
                           'prox sensor 6', 'prox sensor 7'))
    axes[1].grid()
    axes[1].set(ylabel='sensing')

    sns.lineplot(x="timestep", y="myt2_control", data=df_control, ax=axes[2])
    axes[2].set_title('Thymio 2 Control', weight='bold', fontsize=12)
    axes[2].grid()
    axes[2].set(ylabel='control')

    plt.tight_layout()

    filename = 'compare-simulation-seaborn-%s.png' % model
    file = os.path.join(img_dir, filename)
    plt.savefig(file)
    plt.show()


def plot_losses(train_loss, valid_loss, img_dir, title, filename, scale=None):
    """

    :param train_loss: the training losses
    :param valid_loss: the testing losses
    :param img_dir: directory for the output image
    :param title:
    :param filename:
    :param scale:
    """
    x = np.arange(1, len(train_loss) + 1, dtype=int)

    plt.xlabel('epoch', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    plt.xticks(x)

    plt.plot(x, train_loss, label='Train ' + title)
    plt.plot(x, valid_loss, label='Validation ' + title)
    if scale is not None:
        plt.ylim(0, scale)

    plt.legend()
    plt.title(title, weight='bold', fontsize=12)

    check_dir(img_dir)
    file = os.path.join(img_dir, filename)
    plt.savefig(file)
    plt.show()


def my_scatterplot(x, y, x_label, y_label, img_dir, title, filename):
    """
     # Plot a scatter plot with the groundtruth on x-axis and prediction on y-axis
    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :param img_dir:
    :param title:
    :param filename:
    """
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(y_label, fontsize=11)

    plt.scatter(x, y)
    plt.title(title, weight='bold', fontsize=12)

    check_dir(img_dir)
    file = os.path.join(img_dir, filename)
    plt.savefig(file)
    plt.show()


def my_histogram(prediction, x_label, img_dir, title, filename, label=None):
    """
     # plot a scatter plot with the groundtruth on x-axis and prediction on y-axis
    :param prediction:
    :param x_label:
    :param img_dir:
    :param title:
    :param filename:
    :param label:
    """
    plt.xlabel(x_label, fontsize=11)
    if label is None:
        plt.hist(prediction, bins=50)
    else:
        plt.style.use('seaborn-deep')
        plt.hist(prediction, bins=50, label=label)
        plt.legend()
    plt.yscale('log')

    plt.title(title, weight='bold', fontsize=12)

    check_dir(img_dir)
    file = os.path.join(img_dir, filename)
    plt.savefig(file)
    plt.show()


runs_dir = 'out/'
model = '5myts'
img_dir = 'models/distributed/images/dataset/'

visualise_simulation(runs_dir, img_dir, model)
visualise_simulations_comparison(runs_dir, img_dir, model)

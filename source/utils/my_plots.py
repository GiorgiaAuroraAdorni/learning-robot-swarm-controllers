import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression

from utils import utils

sns.set(style="white")


def make_space_above(axes, topmargin=1):
    """
    Increase figure size to make topmargin (in inches) space for titles, without changing the axes sizes
    :param axes:
    :param topmargin:
    """
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1 - s.top) * h + topmargin
    fig.subplots_adjust(bottom=s.bottom * h / figh, top=1 - topmargin / figh)
    fig.set_figheight(figh)


def save_visualisation(filename, img_dir, make_space=False, axes=None):
    """

    :param filename:
    :param img_dir:
    :param make_space:
    :param axes:

    """
    file = os.path.join(img_dir, '%s.pdf' % filename)
    img = os.path.join(img_dir, '%s.png' % filename)
    if make_space:
        make_space_above(axes, topmargin=1)

    plt.savefig(file)
    plt.savefig(img, dpi=300, bbox_inches='tight')
    plt.close()


def plot_distance_from_goal(runs_dir, img_dir, title, filename):
    """
    :param runs_dir:
    :param img_dir:
    :param title
    :param filename
    """

    runs = utils.load_dataset(runs_dir, 'simulation.pkl')
    runs_sub = runs[['timestep', 'goal_position_distance', 'name', 'run']]
    position_distances = get_position_distances(runs_sub)

    max_time_step = runs['timestep'].max()
    time_steps = np.arange(max_time_step)

    myts = ['myt2', 'myt3', 'myt4']

    fig, axes = plt.subplots(ncols=3, figsize=(12.8, 4.8), sharey='col', constrained_layout=True)
    axes[0].set_ylabel('distance from goal', fontsize=11)

    for idx, m in enumerate(myts):
        myt = position_distances[position_distances['name'] == m].drop(columns='name')

        # mean = myt.groupby(['timestep']).mean().squeeze()
        # std = myt.groupby(['timestep']).std().squeeze()
        #
        # ln, = plt.plot(time_steps, mean, label='mean')
        # plt.fill_between(time_steps, mean - std, mean + std, alpha=0.2, label='+/- 1 std',
        #                  color=ln.get_color())

        q1 = myt.groupby(['timestep']).quantile(0.25).squeeze()
        q2 = myt.groupby(['timestep']).quantile(0.75).squeeze()
        q3 = myt.groupby(['timestep']).quantile(0.10).squeeze()
        q4 = myt.groupby(['timestep']).quantile(0.90).squeeze()
        median = myt.groupby(['timestep']).median().squeeze()

        ln, = axes[idx].plot(time_steps, median, label='median')
        axes[idx].fill_between(time_steps, q1, q2, alpha=0.2, label='interquartile range', color=ln.get_color())
        axes[idx].fill_between(time_steps, q3, q4, alpha=0.1, label='interdecile range', color=ln.get_color())
        axes[idx].set_xlabel('timestep', fontsize=11)
        axes[idx].set_ylim(top=10)
        axes[idx].set_title(m, fontsize=12)

    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles=handles, labels=labels, loc='lower center', fontsize=11, bbox_to_anchor=(0.5, -0.3), ncol=3,
               bbox_transform=axes[1].transAxes)
    fig.suptitle(title, weight='bold', fontsize=12)
    save_visualisation(filename, img_dir)


def get_position_distances(runs_sub, with_run=False):
    """

    :param runs_sub:
    :return position_distances:
    """
    v = utils.cartesian_product(runs_sub.timestep.unique(), runs_sub.run.unique(), runs_sub.name.unique())
    idx = pd.MultiIndex.from_arrays([v[:, 0], v[:, 1], v[:, 2]])

    position_distances = runs_sub.set_index(['timestep', 'run', 'name']).reindex(idx)
    position_distances.index.names = ['timestep', 'run', 'name']
    position_distances = position_distances.reset_index()

    if not with_run:
        position_distances = position_distances.fillna(0).drop(columns='run')

    return position_distances


def plot_compared_distance_from_goal(dataset_folders, img_dir, title, filename):
    """

    :param dataset_folders:
    :param img_dir:
    :param title:
    :param filename:
    """

    utils.check_dir(img_dir)
    datasets = ['omniscient', 'manual', 'distributed', 'communication']

    positions = []
    timesteps = []

    for el in dataset_folders:

        runs = utils.load_dataset(el, 'simulation.pkl')
        runs_sub = runs[['timestep', 'goal_position_distance', 'name', 'run']]
        position_distances = get_position_distances(runs_sub)
        max_time_step = runs['timestep'].max()
        time_steps = np.arange(max_time_step)

        positions.append(position_distances)
        timesteps.append(time_steps)

    myts = ['myt2', 'myt3', 'myt4']

    fig, axes = plt.subplots(ncols=3, figsize=(12.8, 4.8), sharey='col', constrained_layout=True)
    axes[0].set_ylabel('distance from goal', fontsize=11)

    for idx, m in enumerate(myts):
        for d_idx, d in enumerate(datasets):
            position_distances = positions[d_idx]
            time_steps = timesteps[d_idx]

            myt = position_distances[position_distances['name'] == m].drop(columns='name')

            q1 = myt.groupby(['timestep']).quantile(0.25).squeeze()
            q2 = myt.groupby(['timestep']).quantile(0.75).squeeze()
            q3 = myt.groupby(['timestep']).quantile(0.10).squeeze()
            q4 = myt.groupby(['timestep']).quantile(0.90).squeeze()
            median = myt.groupby(['timestep']).median().squeeze()

            ln, = axes[idx].plot(time_steps, median, label='median (%s)' % d)
            axes[idx].fill_between(time_steps, q1, q2, alpha=0.2, label='interquartile range (%s)' % d, color=ln.get_color())
            axes[idx].fill_between(time_steps, q3, q4, alpha=0.1, label='interdecile range (%s)' % d, color=ln.get_color())

            axes[idx].set_xlabel('timestep', fontsize=11)
            axes[idx].set_xlim(0, 17)
            axes[idx].set_ylim(0, 10)

            xint = range(0, max_time_step + 1, max_time_step // 6)
            axes[idx].set_xticks(xint)

            axes[idx].set_title(m, fontsize=12)

    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()

    handles = [handles[0], handles[1], handles[2], handles[3],
               handles[4], handles[6], handles[8], handles[10],
               handles[5], handles[7], handles[9], handles[11]]
    labels = [labels[0], labels[1], labels[2], labels[3],
              labels[4], labels[6], labels[8], labels[10],
              labels[5], labels[7], labels[9], labels[11]]

    fig.legend(handles=handles, labels=labels, loc='lower center', fontsize=11, bbox_to_anchor=(0.5, -0.5), ncol=3,
               bbox_transform=axes[1].transAxes)

    fig.suptitle(title, weight='bold', fontsize=12)
    save_visualisation(filename, img_dir)


def visualise_simulation(runs_dir, img_dir, simulation, title, net_input):
    """

    :param runs_dir:
    :param img_dir:
    :param simulation:
    :param title:
    :param net_input:
    """
    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]

    run = runs_sub[runs_sub['run'] == simulation]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    max_time_step = run['timestep'].max()
    time_steps = np.arange(max_time_step)

    run_myt2 = run[run['name'] == 'myt2'].drop(columns='name').reset_index()
    # FIXME
    _, myt2_control, _, run_myt2, proximity_sensors = utils.extract_input_output(run_myt2, net_input, N=1)
    myt2_sensing = run_myt2[proximity_sensors]

    plt.figure()
    fig, axes = plt.subplots(nrows=3, figsize=(7, 11), sharex=True)

    # Plot the evolution of the positions of all robots over time
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    for i, name in enumerate(run.name.unique()):
        x = np.array(run[run['name'] == name].apply(lambda row: list(row.position)[0], axis=1))
        axes[0].plot(time_steps, x, label='myt%d' % (i + 1))
    axes[0].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23),
                   ncol=len(run.name.unique()), title="robot")
    axes[0].set_yticks(target)
    axes[0].grid()

    # Plot, for a given robot, the evolution of sensing over time
    axes[1].set_ylabel('sensing', fontsize=11)
    axes[1].set_title('Thymio 2 Sensing', weight='bold', fontsize=12)
    for s in proximity_sensors:
        axes[1].plot(time_steps, myt2_sensing[s], label=s)
    axes[1].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23),
                   ncol=len(proximity_sensors), title="proximity sensor")
    axes[1].grid()

    # Plot, for a given robot, the evolution of control over time
    axes[2].set_ylabel('control', fontsize=11)
    axes[2].set_title('Thymio 2 Control', weight='bold', fontsize=12)
    axes[2].plot(time_steps, myt2_control, color='black')
    axes[2].grid()

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    plt.xlabel('timestep', fontsize=11)
    fig.suptitle(title, fontsize=14, weight='bold')

    filename = 'plot-simulation-%d' % simulation
    save_visualisation(filename, img_dir, make_space=True, axes=axes)


def visualise_simulation_all_sensors(runs_dir, img_dir, simulation, title, net_input):
    """

    :param runs_dir:
    :param img_dir:
    :param simulation:
    :param title:
    :param net_input:
    """
    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]

    run = runs_sub[runs_sub['run'] == simulation]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    max_time_step = run['timestep'].max()
    time_steps = np.arange(max_time_step)

    run_myt2 = run[run['name'] == 'myt2'].drop(columns='name').reset_index()
    # FIXME
    _, myt2_control, _, run_myt2, proximity_sensors = utils.extract_input_output(run_myt2, net_input, N=1)

    plt.figure(constrained_layout=True)
    fig, axes = plt.subplots(nrows=2, figsize=(6, 7), sharex=True)

    # Plot the evolution of the positions of all robots over time
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    for i, name in enumerate(run.name.unique()):
        x = np.array(run[run['name'] == name].apply(lambda row: list(row.position)[0], axis=1))
        axes[0].plot(time_steps, x, label='myt%d' % (i + 1))

    if not len(run.name.unique()) > 5:
        col = len(run.name.unique())
        pos = (0.5, -0.23)
    else:
        col = int(np.ceil(len(run.name.unique())/2))
        pos = (0.5, -0.3)

    axes[0].legend(loc='lower center', fontsize='small', bbox_to_anchor=pos,
                   ncol=col, title="robot")
    axes[0].set_yticks(target)
    axes[0].grid()

    # Plot, for a given robot, the evolution of control over time
    axes[1].set_ylabel('control', fontsize=11)
    axes[1].set_ylim(-18, 18)
    axes[1].set_title('Thymio 2 Control', weight='bold', fontsize=12)
    axes[1].plot(time_steps, myt2_control, color='black')
    axes[1].grid()

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    plt.xlabel('timestep', fontsize=11)
    fig.suptitle(title, fontsize=14, weight='bold')

    filename = 'plot-simulation-%d' % simulation
    save_visualisation(filename, img_dir, make_space=True, axes=axes)


def visualise_simulations_comparison(runs_dir, img_dir, title, net_input):
    """
    :param runs_dir:
    :param img_dir:
    :param title
    :param net_input:
    """
    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]
    runs['x_position'] = runs.apply(lambda row: list(row.position)[0], axis=1)

    max_time_step = runs_sub['timestep'].max()
    time_steps = np.arange(max_time_step)

    run = runs_sub[runs_sub['run'] == 0]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    runs_myt2 = runs[runs['name'] == 'myt2'].drop(columns='name').reset_index()
    # FIXME
    _, _, _, runs_myt2, proximity_sensors = utils.extract_input_output(runs_myt2, net_input, N=1)

    plt.figure()
    fig, axes = plt.subplots(nrows=3, figsize=(7, 11), sharex=True)

    # Plot the evolution of the positions of all robots over time
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)

    for name in runs.name.unique():
        runs_myt = runs[runs['name'] == name].reset_index()
        mean_x_positions = np.array(runs_myt.groupby('timestep').x_position.mean())
        std_x_positions = np.array(runs_myt.groupby('timestep').x_position.std())

        axes[0].plot(time_steps, mean_x_positions, label=name)
        axes[0].fill_between(time_steps, mean_x_positions - std_x_positions,
                             mean_x_positions + std_x_positions, alpha=0.2)

    axes[0].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23),
                   ncol=len(runs.name.unique()), title="robot")
    axes[0].set_yticks(target)
    axes[0].grid()

    # Plot, for a given robot, the evolution of sensing over time
    axes[1].set_ylabel('sensing', fontsize=11)
    axes[1].set_title('Thymio 2 Sensing', weight='bold', fontsize=12)
    for s in proximity_sensors:
        mean_myt2_sensing = np.array(runs_myt2.groupby('timestep')[s].mean())
        std_myt2_sensing = np.array(runs_myt2.groupby('timestep')[s].std())

        axes[1].plot(time_steps, mean_myt2_sensing, label=s)  # , color='black')
        axes[1].fill_between(time_steps,
                             mean_myt2_sensing - std_myt2_sensing,
                             mean_myt2_sensing+ std_myt2_sensing,
                             alpha=0.2)
    axes[1].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23),
                   ncol=len(proximity_sensors), title="proximity sensor")
    axes[1].grid()

    # Plot, for a given robot, the evolution of control over time
    axes[2].set_ylabel('control', fontsize=11)
    axes[2].set_title('Thymio 2 Control', weight='bold', fontsize=12)

    mean_myt2_control = np.array(runs_myt2.groupby('timestep').motor_left_target.mean())
    std_myt2_control = np.array(runs_myt2.groupby('timestep').motor_left_target.std())

    axes[2].plot(time_steps, mean_myt2_control, color='black')
    axes[2].fill_between(time_steps,
                         mean_myt2_control - std_myt2_control,
                         mean_myt2_control + std_myt2_control,
                         alpha=0.2)
    axes[2].grid()

    plt.xlabel('timestep', fontsize=11)
    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    filename = 'compare-simulation'
    save_visualisation(filename, img_dir, make_space=True, axes=axes)


def visualise_simulations_comparison_all_sensors(runs_dir, img_dir, title, net_input):
    """
    :param runs_dir:
    :param img_dir:
    :param title
    :param net_input:
    """
    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]
    runs['x_position'] = runs.apply(lambda row: list(row.position)[0], axis=1)

    max_time_step = runs_sub['timestep'].max()
    time_steps = np.arange(max_time_step)

    run = runs_sub[runs_sub['run'] == 0]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    runs_myt2 = runs[runs['name'] == 'myt2'].drop(columns='name').reset_index()
    # FIXME
    _, _, _, runs_myt2, proximity_sensors = utils.extract_input_output(runs_myt2, net_input, N=1)

    plt.figure(constrained_layout=True)
    fig, axes = plt.subplots(nrows=2, figsize=(6, 7), sharex=True)

    # Plot the evolution of the positions of all robots over time
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    for name in runs.name.unique():
        runs_myt = runs[runs['name'] == name].reset_index()
        mean_x_positions = np.array(runs_myt.groupby('timestep').x_position.mean())
        std_x_positions = np.array(runs_myt.groupby('timestep').x_position.std())

        axes[0].plot(time_steps, mean_x_positions, label=name)
        axes[0].fill_between(time_steps, mean_x_positions - std_x_positions,
                             mean_x_positions + std_x_positions, alpha=0.2)

    if not len(run.name.unique()) > 5:
        col = len(run.name.unique())
        pos = (0.5, -0.23)
    else:
        col = int(np.ceil(len(run.name.unique())/2))
        pos = (0.5, -0.3)

    axes[0].legend(loc='lower center', fontsize='small', bbox_to_anchor=pos,
                   ncol=col, title="robot")

    axes[0].set_yticks(target)
    axes[0].grid()

    # Plot, for a given robot, the evolution of control over time
    axes[1].set_ylabel('control', fontsize=11)
    axes[1].set_title('Thymio 2 Control', weight='bold', fontsize=12)

    mean_myt2_control = np.array(runs_myt2.groupby('timestep').motor_left_target.mean())
    std_myt2_control = np.array(runs_myt2.groupby('timestep').motor_left_target.std())

    axes[1].plot(time_steps, mean_myt2_control, color='black')
    axes[1].fill_between(time_steps,
                         mean_myt2_control - std_myt2_control,
                         mean_myt2_control + std_myt2_control,
                         alpha=0.2)
    axes[1].grid()

    plt.xlabel('timestep', fontsize=11)
    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    filename = 'compare-simulation'
    save_visualisation(filename, img_dir, make_space=True, axes=axes)


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

    plt.figure()
    plt.xlabel('epoch', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    plt.plot(x, train_loss, label='train')
    plt.plot(x, valid_loss, label='validation')

    plt.ylim(0, max(min(train_loss), min(valid_loss)) + 50)

    xint = range(0, math.ceil(max(x)) + 1, max(x)//10)
    plt.xticks(xint)

    plt.legend()
    plt.title(title, weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def my_scatterplot(x, y, x_label, y_label, img_dir, title, filename):
    """
     Plot a scatter plot. Usually with the groundtruth on x-axis and prediction on y-axis
    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :param img_dir:
    :param title:
    :param filename:
    """
    plt.figure()
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(y_label, fontsize=11)

    plt.scatter(x, y, alpha=0.5, marker='.')
    plt.title(title, weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def my_histogram(prediction, x_label, img_dir, title, filename, label=None):
    """
    
    :param prediction:
    :param x_label:
    :param img_dir:
    :param title:
    :param filename:
    :param label:
    """
    plt.figure(constrained_layout=True)

    plt.yscale('log')
    plt.xlabel(x_label, fontsize=11)

    plt.title(title, weight='bold', fontsize=12)

    if label is None:
        plt.hist(prediction)
    else:
        plt.hist(prediction, label=label)
        plt.legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.32), ncol=len(label),
                   title="proximity sensor", title_fontsize=11, markerscale=0.2)

    save_visualisation(filename, img_dir)


def plot_target_distribution(y_g, y_p, img_dir, title, filename):
    """
    
    :param y_g:
    :param y_p:
    :param img_dir:
    :param title
    :param filename:
    """
    labels = ['groundtruth', 'prediction']

    plt.figure(constrained_layout=True)
    plt.yscale('log')

    y = np.array([y_g, y_p]).reshape(-1, 2)
    plt.hist(y, bins=50, label=labels)
    plt.legend()

    plt.title(title, weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def plot_regressor(x, y, x_label, y_label, img_dir, title, filename):
    """

    :param x:
    :param y:
    :param x_label:
    :param y_label:
    :param img_dir:
    :param title:
    :param filename:
    """
    lr = LinearRegression()
    lr.fit(np.reshape(x, [-1, 1]), np.reshape(y, [-1, 1]))
    score = lr.score(np.reshape(x, [-1, 1]), np.reshape(y, [-1, 1]))

    plt.figure()

    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(y_label, fontsize=11)

    plt.scatter(x, y, alpha=0.3, marker='.', label='sample')
    y_hat = lr.predict(np.reshape(x, [-1, 1]))
    if len(y_hat.shape) > 1:
        y_hat = y_hat.squeeze()
    plt.plot(x, y_hat, color="orange", label='regression: $R^2=%.3f$' % score)

    if y_label == 'transmitted_comm':
        plt.ylim(0, 1)

    plt.title(title, weight='bold', fontsize=12)
    plt.legend()

    save_visualisation(filename, img_dir)


def plot_response(x, y, x_label, img_dir, title, filename, index=None):
    """

    :param x
    :param y
    :param x_label
    :param img_dir
    :param title
    :param filename
    :param index: this parameter is different from None only when x is the input sensing, otherwise, x is a 1D vector
    """

    if index is not None:
        x = np.multiply(x[:, index], 1000)
        # y = y[0]

    plt.figure()
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel('control', fontsize=11)

    plt.plot(x.tolist(), y)

    plt.title(title, weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def plot_sensing_timestep(runs_dir, img_dir, net_input, model):
    """

    :param runs_dir
    :param img_dir
    :param net_input
    :param model
    """
    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['timestep', 'run', 'prox_values', 'prox_comm', 'all_sensors', 'motor_left_target']]

    max_time_step = runs_sub['timestep'].max()
    time_steps = np.arange(max_time_step)
    # FIXME
    _, _, _, runs_, proximity_sensors = utils.extract_input_output(runs_sub, net_input, N=1)

    # Mean of the sensing of each run, among all the robots
    plt.figure(constrained_layout=True)

    # Plot the evolution of the sensing over time
    plt.ylabel('sensing (%s)' % net_input, fontsize=11)

    for s in proximity_sensors:
        mean_sensing = np.array(runs_.groupby('timestep')[s].mean())
        std_sensing = np.array(runs_.groupby('timestep')[s].std())

        plt.plot(time_steps, mean_sensing, label=s)
        plt.fill_between(time_steps,
                         mean_sensing - std_sensing,
                         mean_sensing + std_sensing,
                         alpha=0.2)

    plt.legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.3), ncol=len(proximity_sensors),
               title="proximity sensor")

    plt.xlabel('timestep', fontsize=11)
    title = 'Response Sensing %s' % model
    plt.title(title, weight='bold', fontsize=12)

    file_name = 'response-sensing-%s' % model
    save_visualisation(file_name, img_dir)


def visualise_communication_simulation(runs_dir, img_dir, simulation, title):
    """

    :param runs_dir:
    :param img_dir:
    :param simulation:
    :param title:
    """
    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'transmitted_comm']]

    run = runs_sub[runs_sub['run'] == simulation]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    max_time_step = run['timestep'].max()
    time_steps = np.arange(max_time_step)

    fig = plt.figure(figsize=(8.8, 4.8))
    norm = plt.Normalize(0, 1)

    # Plot the evolution of the positions of all robots over time
    for i, name in enumerate(run.name.unique()):
        run_myt = run[run['name'] == name].reset_index()
        x = np.array(run[run['name'] == name].apply(lambda row: list(row.position)[0], axis=1))

        comm = np.array(run_myt.transmitted_comm, dtype='float32')
        comm = np.divide(comm, 2 ** 10)

        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be (numlines) x (points per line) x 2 (for x and y)
        points = np.array([time_steps, x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(comm)
        # lc.set_linewidth(2)

        ax = fig.gca()
        line = ax.add_collection(lc)
        line.set_clim(0, 1)

    plt.colorbar(line, ax=ax, boundaries=np.linspace(0, 1, 50), ticks=np.linspace(0, 1, 5))
    if max_time_step > 1:
        plt.xlim(0, max_time_step - 1)
    plt.ylim(-1, target[4] + 1)

    plt.yticks(target)
    plt.grid()

    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('x position', fontsize=11)

    plt.tight_layout()
    plt.title(title, fontsize=12, weight='bold')

    filename = 'plot-simulation-communication-%d' % simulation
    save_visualisation(filename, img_dir)


def plot_compared_distance_compressed(dataset_folders, img_dir, datasets, title, filename, absolute=True):
    """

    :param dataset_folders:
    :param img_dir:
    :param datasets
    :param title:
    :param filename:
    :param absolute
    """

    utils.check_dir(img_dir)

    positions = []
    timesteps = []

    for el in dataset_folders:

        runs = utils.load_dataset(el, 'simulation.pkl')
        if absolute:
            runs_sub = runs[['timestep', 'goal_position_distance_absolute', 'name', 'run']]
        else:
            runs_sub = runs[['timestep', 'goal_position_distance', 'name', 'run']]

        position_distances = get_position_distances(runs_sub, with_run=True)
        max_time_step = runs_sub['timestep'].max()
        time_steps = np.arange(max_time_step)

        positions.append(position_distances)
        timesteps.append(time_steps)

    fig = plt.figure()
    plt.ylabel('distance from goal', fontsize=11)
    plt.xlabel('timestep', fontsize=11)

    for d_idx, d in enumerate(datasets):
        if absolute:
            position_distances = positions[d_idx].groupby('timestep').goal_position_distance_absolute
        else:
            position_distances = positions[d_idx].groupby('timestep').goal_position_distance

        q1 = position_distances.quantile(0.25).squeeze()
        q2 = position_distances.quantile(0.75).squeeze()
        q3 = position_distances.quantile(0.10).squeeze()
        q4 = position_distances.quantile(0.90).squeeze()
        median = position_distances.median().squeeze()

        ln, = plt.plot(timesteps[d_idx], median, label='median (%s)' % d)
        plt.fill_between(timesteps[d_idx], q1, q2, alpha=0.2, label='interquartile range (%s)' % d, color=ln.get_color())
        plt.fill_between(timesteps[d_idx], q3, q4, alpha=0.1, label='interdecile range (%s)' % d, color=ln.get_color())

    plt.xlim(0, time_steps.max())
    # FIXME depends if it is used goal_position_distance or goal_position_distance_absolute
    # plt.ylim(0, 10)

    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()

    if len(datasets) > 1:
        handles = [handles[0], handles[1], handles[2], handles[3],
                   handles[4], handles[6], handles[8], handles[10],
                   handles[5], handles[7], handles[9], handles[11]]
        labels = [labels[0], labels[1], labels[2], labels[3],
                  labels[4], labels[6], labels[8], labels[10],
                  labels[5], labels[7], labels[9], labels[11]]

        plt.legend(handles=handles, labels=labels, loc='lower center', fontsize=11, bbox_to_anchor=(0.5, -0.5), ncol=3)

    plt.legend(handles=handles, labels=labels, fontsize=11)

    plt.title(title, weight='bold', fontsize=12)
    save_visualisation(filename, img_dir)


def visualise_communication_vs_control(runs_dir, img_dir, title):
    """

    :param runs_dir:
    :param img_dir:
    :param title:
    """

    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['motor_left_target', 'transmitted_comm']]

    x = np.array(runs_sub.motor_left_target)
    y = np.divide(np.array(runs_sub.transmitted_comm), (2 ** 10))
    plot_regressor(x, y, 'motor_left_target', 'transmitted_comm', img_dir, title,
                   'regression-control-communication')


def visualise_communication_vs_distance(runs_dir, img_dir, title):
    """

    :param runs_dir:
    :param img_dir:
    :param title:
    """

    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['position', 'goal_position', 'goal_position_distance', 'transmitted_comm']]

    runs_sub[['x_position', 'y_position']] = pd.DataFrame(runs_sub.position.tolist(), index=runs_sub.index)
    runs_sub[['x_goal_position', 'y_goal_position']] = pd.DataFrame(runs_sub.goal_position.tolist(), index=runs_sub.index)

    x1 = np.array(runs_sub.x_goal_position - runs_sub.x_position)
    x = np.array(runs_sub.goal_position_distance)
    y = np.divide(np.array(runs_sub.transmitted_comm), (2 ** 10))

    plot_regressor(x, y, 'absolute_distance_from_goal', 'transmitted_comm', img_dir, title,
                   'regression-absolute-distance-communication')

    plot_regressor(x1, y, 'distance_from_goal', 'transmitted_comm', img_dir, title,
                   'regression-distance-communication')


def visualise_simulation_over_time_all_sensors(runs_dir, img_dir, simulation, title):
    """

    :param runs_dir:
    :param img_dir:
    :param simulation:
    :param title:
    """
    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]

    run = runs_sub[runs_sub['run'] == simulation]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    max_time_step = run['timestep'].max()
    time_steps = np.arange(max_time_step)

    # Plot the evolution of the positions of all robots over time
    fig = plt.figure(constrained_layout=True)
    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('x position', fontsize=11)
    plt.yticks(target)
    plt.grid()

    plt.title(title, weight='bold', fontsize=12)

    for i, name in enumerate(run.name.unique()):
        x = np.array(run[run['name'] == name].apply(lambda row: list(row.position)[0], axis=1))
        plt.plot(time_steps, x, label='myt%d' % (i + 1))

    if not len(run.name.unique()) > 5:
        col = len(run.name.unique())
        pos = (0.5, -0.3)
    else:
        col = int(np.ceil(len(run.name.unique())/2))
        pos = (0.5, -0.4)

    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles=handles, labels=labels, loc='lower center', fontsize='small', bbox_to_anchor=pos, ncol=col, title="robot")

    filename = 'plot-simulation-%d' % simulation
    save_visualisation(filename, img_dir)


def thymio_quantity_distribution(runs_dir, img_dir, title, filename):
    """

    :param runs_dir:
    :param img_dir:
    :param title:
    :param filename:
    """
    runs = utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['run', 'myt_quantity']]

    df = runs_sub.drop_duplicates()

    plt.figure()
    plt.ylabel('simulations', fontsize=11)
    plt.xlabel('number of thymios', fontsize=11)
    plt.title(title, weight='bold', fontsize=12)

    plt.hist(df.myt_quantity, bins=np.linspace(4.5, 10.5, 7), rwidth=0.8)
    save_visualisation(filename, img_dir)

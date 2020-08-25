import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression

import utils

sns.set(style="white")


def make_space_above(axes, topmargin=1):
    """
    Increase figure size to make topmargin (in inches) space for titles, without changing the axes sizes.

    :param axes: axes of the image
    :param topmargin: topmargin
    """
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1 - s.top) * h + topmargin
    fig.subplots_adjust(bottom=s.bottom * h / figh, top=1 - topmargin / figh)
    fig.set_figheight(figh)


def save_visualisation(filename, img_dir, make_space=False, axes=None):
    """

    :param filename: name of the image
    :param img_dir: path where to save the image
    :param make_space: if make space above the image
    :param axes: axes of the image
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
    
    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param title: title of the image
    :param filename: name of the image
    """

    runs = utils.utils.load_dataset(runs_dir, 'simulation.pkl')
    runs_sub = runs[['timestep', 'goal_position_distance', 'name', 'run']]
    position_distances = get_position_distances(runs_sub)

    max_time_step = runs['timestep'].max()
    time_steps = np.arange(max_time_step)

    myts = ['myt2', 'myt3', 'myt4']

    fig, axes = plt.subplots(ncols=3, figsize=(12.8, 4.8), sharey='col', constrained_layout=True)
    axes[0].set_ylabel('distance from goal', fontsize=11)

    for idx, m in enumerate(myts):
        myt = position_distances[position_distances['name'] == m].drop(columns='name')
        
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

    :param runs_sub: directory containing the simulation runs 
    :param with_run: boolean, default False
    :return position_distances: array containing distances from goal
    """
    v = utils.utils.cartesian_product(runs_sub.timestep.unique(), runs_sub.run.unique(), runs_sub.name.unique())
    idx = pd.MultiIndex.from_arrays([v[:, 0], v[:, 1], v[:, 2]])

    position_distances = runs_sub.set_index(['timestep', 'run', 'name']).reindex(idx)
    position_distances.index.names = ['timestep', 'run', 'name']
    position_distances = position_distances.reset_index()

    if not with_run:
        position_distances = position_distances.fillna(0).drop(columns='run')

    return position_distances


def plot_compared_distance_from_goal(dataset_folders, img_dir, title, filename, absolute=True):
    """

    :param dataset_folders: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param title: title of the image 
    :param filename:  name of the image 
    :param absolute: boolean value that states is use absolute distances from goal (default: True)
    """

    utils.utils.check_dir(img_dir)
    datasets = ['omniscient', 'manual', 'distributed', 'distributed (control * 5)', 'communication', 'communication (control * 5)']

    positions = []
    timesteps = []
    max_time_step = None

    for el in dataset_folders:
        runs = utils.utils.load_dataset(el, 'simulation.pkl')
        if absolute:
            runs_sub = runs[['timestep', 'goal_position_distance_absolute', 'name', 'run']]
        else:
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
            # axes[idx].set_xlim(0, 36)
            # axes[idx].set_ylim(0, 10)

            xint = range(0, max_time_step + 1, max_time_step // 6)
            axes[idx].set_xticks(xint)

            axes[idx].set_title(m, fontsize=12)

    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()

    if len(dataset_folders) > 1:
        handles = [handles[0], handles[1], handles[2], handles[3],
                   handles[4], handles[6], handles[8], handles[10],
                   handles[5], handles[7], handles[9], handles[11]]
        labels = [labels[0], labels[1], labels[2], labels[3],
                  labels[4], labels[6], labels[8], labels[10],
                  labels[5], labels[7], labels[9], labels[11]]

        fig.legend(handles=handles, labels=labels, loc='lower center', fontsize=11,
                   bbox_to_anchor=(0.5, -0.5), ncol=3, bbox_transform=axes[1].transAxes)
    else:
        fig.legend(handles=handles, labels=labels, fontsize=11)

    fig.suptitle(title, weight='bold', fontsize=12)
    save_visualisation(filename, img_dir)


def visualise_simulation(runs_dir, img_dir, simulation, title, net_input):
    """

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param simulation: simulation to use
    :param title: title of the image 
    :param net_input: input of the net between prox_values, prox_comm or all_sensors  
    """
    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]

    run = runs_sub[runs_sub['run'] == simulation]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    max_time_step = run['timestep'].max()
    time_steps = np.arange(max_time_step)

    run_myt2 = run[run['name'] == 'myt2'].drop(columns='name').reset_index()
    _, myt2_control, _, run_myt2, proximity_sensors = utils.utils.extract_input_output(run_myt2, net_input, N=1)
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

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param simulation: simultion to use
    :param title: title of the image 
    :param net_input: input of the net between prox_values, prox_comm or all_sensors
    """
    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]

    run = runs_sub[runs_sub['run'] == simulation]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    max_time_step = run['timestep'].max()
    time_steps = np.arange(max_time_step)

    run_myt2 = run[run['name'] == 'myt2'].drop(columns='name').reset_index()
    _, myt2_control, _, run_myt2, proximity_sensors = utils.utils.extract_input_output(run_myt2, net_input, N=1)

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

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param title: title of the image
    :param net_input: input of the net between prox_values, prox_comm or all_sensors  
    """
    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]
    runs['x_position'] = runs.apply(lambda row: list(row.position)[0], axis=1)

    max_time_step = runs_sub['timestep'].max()
    time_steps = np.arange(max_time_step)

    run = runs_sub[runs_sub['run'] == 0]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    runs_myt2 = runs[runs['name'] == 'myt2'].drop(columns='name').reset_index()
    _, _, _, runs_myt2, proximity_sensors = utils.utils.extract_input_output(runs_myt2, net_input, N=1)

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
                             mean_myt2_sensing + std_myt2_sensing,
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
    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param title: title of the image
    :param net_input: input of the net between prox_values, prox_comm or all_sensors  
    """
    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['name', 'timestep', 'run', 'position', 'goal_position', 'motor_left_target', 'prox_values',
                     'prox_comm', 'all_sensors']]
    runs['x_position'] = runs.apply(lambda row: list(row.position)[0], axis=1)

    max_time_step = runs_sub['timestep'].max()
    time_steps = np.arange(max_time_step)

    run = runs_sub[runs_sub['run'] == 0]
    target = np.array(run[run['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

    runs_myt2 = runs[runs['name'] == 'myt2'].drop(columns='name').reset_index()
    _, _, _, runs_myt2, proximity_sensors = utils.utils.extract_input_output(runs_myt2, net_input, N=1)

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


def plot_losses(train_loss, valid_loss, img_dir, title, filename, goal):
    """

    :param train_loss: the training losses
    :param valid_loss: the testing losses
    :param img_dir: directory for the output image
    :param title: title of the image 
    :param filename: name of the image
    """
    x = np.arange(1, len(train_loss) + 1, dtype=int)

    plt.figure()
    plt.xlabel('epoch', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    plt.plot(x, train_loss, label='train')
    plt.plot(x, valid_loss, label='validation')

    if goal == 'colour':
        plt.ylim(0, 1)
    else:
        plt.ylim(0, max(min(train_loss), min(valid_loss)) + 50)

    xint = range(1, np.ceil(max(x)).astype(int) + 1, np.ceil(max(x)/10).astype(int))
    plt.xticks(xint)

    plt.legend()
    plt.title(title, weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def my_scatterplot(x, y, x_label, y_label, img_dir, title, filename):
    """
     Plot a scatter plot. Usually with the groundtruth on x-axis and prediction on y-axis.

    :param x: values for the x-axis
    :param y: values for the y-axis
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param img_dir: directory containing the simulation images
    :param title: title of the image 
    :param filename: name of the image
    """
    plt.figure()
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(y_label, fontsize=11)

    plt.scatter(x, y, alpha=0.5, marker='.')
    plt.title(title, weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def my_histogram(prediction, x_label, img_dir, title, filename, label=None):
    """
    
    :param prediction: predictions
    :param x_label: label for the x-axis
    :param img_dir: directory containing the simulation images
    :param title: title of the image 
    :param filename: name of the image
    :param label: label for the plot, default None
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
    
    :param y_g: validation groundtruth
    :param y_p: validation prediction
    :param img_dir: directory containing the simulation images
    :param title: title of the image
    :param filename: name of the image
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

    :param x: values for the x-axis
    :param y: values for the y-axis
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param img_dir: directory containing the simulation images
    :param title: title of the image 
    :param filename: name of the image
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


def plot_response(x, y, x_label, img_dir, title, filename, index=None, y_label='control'):
    """

    :param x: values for the x-axis
    :param y: values for the y-axis
    :param x_label: label for the x-axis
    :param img_dir: directory containing the simulation images
    :param title: title of the image
    :param filename: name of the image
    :param index: this parameter is different from None only when x is the input sensing, otherwise, x is a 1D vector
    :param y_label: label for the y-axis (optional, default: 'control')
    """

    if index is not None:
        x = np.multiply(x[:, index], 1000)
        # y = y[0]

    plt.figure()
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(y_label, fontsize=11)

    plt.plot(x.tolist(), y)

    plt.title(title, weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def plot_sensing_timestep(runs_dir, img_dir, net_input, model):
    """

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param net_input: input of the net between prox_values, prox_comm or all_sensors
    :param model: model to be used
    """
    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['timestep', 'run', 'prox_values', 'prox_comm', 'all_sensors', 'motor_left_target']]

    max_time_step = runs_sub['timestep'].max()
    time_steps = np.arange(max_time_step)
    _, _, _, runs_, proximity_sensors = utils.utils.extract_input_output(runs_sub, net_input, N=1)

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

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param simulation: simulation to use
    :param title: title of the image 
    """
    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
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

    .. warning::
        Limits on x and y axes have not yet been fixed.
        Their value depends if it is used ``goal_position_distance`` or
        ``goal_position_distance_absolute``.


    :param dataset_folders: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param datasets: names of the datasets to be uses
    :param title: title of the image 
    :param filename: name of the image
    :param absolute: boolean value that states is use absolute distances from goal (default: True)
    """

    utils.utils.check_dir(img_dir)

    positions = []
    timesteps = []
    max_timestep = 0

    for el in dataset_folders:

        runs = utils.utils.load_dataset(el, 'simulation.pkl')
        if absolute:
            runs_sub = runs[['timestep', 'goal_position_distance_absolute', 'name', 'run']]
        else:
            runs_sub = runs[['timestep', 'goal_position_distance', 'name', 'run']]

        position_distances = get_position_distances(runs_sub, with_run=True)
        max_time_step = runs_sub['timestep'].max()
        if max_time_step > max_timestep:
            max_timestep = max_time_step

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

        median = np.pad(median, ((0, max_timestep - len(median))), mode='edge')
        q1 = np.pad(q1, ((0, max_timestep - len(q1))), mode='edge')
        q2 = np.pad(q2, ((0, max_timestep - len(q2))), mode='edge')
        q3 = np.pad(q3, ((0, max_timestep - len(q3))), mode='edge')
        q4 = np.pad(q4, ((0, max_timestep - len(q4))), mode='edge')

        ln, = plt.plot(np.arange(max_timestep), median, label='median (%s)' % d)
        plt.fill_between(np.arange(max_timestep), q1, q2, alpha=0.2, label='interquartile range (%s)' % d, color=ln.get_color())
        plt.fill_between(np.arange(max_timestep), q3, q4, alpha=0.1, label='interdecile range (%s)' % d, color=ln.get_color())

    # FIXME
    # plt.xlim(0, 17)
    # plt.ylim(0, 10)

    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()

    if len(dataset_folders) > 1:
        handles = [handles[0], handles[1], handles[2], handles[3],
                   handles[4], handles[6], handles[8], handles[10],
                   handles[5], handles[7], handles[9], handles[11]]
        labels = [labels[0], labels[1], labels[2], labels[3],
                  labels[4], labels[6], labels[8], labels[10],
                  labels[5], labels[7], labels[9], labels[11]]

        plt.legend(handles=handles, labels=labels, loc='lower center', fontsize=11, bbox_to_anchor=(0.5, -0.5), ncol=3)
    else:
        plt.legend(handles=handles, labels=labels, fontsize=11)

    plt.title(title, weight='bold', fontsize=12)
    save_visualisation(filename, img_dir)


def visualise_communication_vs_control(runs_dir, img_dir, title):
    """

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param title: title of the image 
    """

    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['motor_left_target', 'transmitted_comm']]

    x = np.array(runs_sub.motor_left_target)
    y = np.divide(np.array(runs_sub.transmitted_comm), (2 ** 10))
    plot_regressor(x, y, 'motor_left_target', 'transmitted_comm', img_dir, title,
                   'regression-control-communication')


def visualise_communication_vs_distance(runs_dir, img_dir, title):
    """

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param title: title of the image 
    """

    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
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

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param simulation: simulation to use
    :param title: title of the image 
    """
    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
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

    :param runs_dir: directory containing the simulation runs
    :param img_dir: directory containing the simulation images
    :param title: title of the image 
    :param filename: name of the image
    """
    runs = utils.utils.load_dataset(runs_dir, 'complete-simulation.pkl')
    runs_sub = runs[['run', 'myt_quantity']]

    df = runs_sub.drop_duplicates()

    plt.figure()
    plt.ylabel('simulations', fontsize=11)
    plt.xlabel('number of thymios', fontsize=11)
    plt.title(title, weight='bold', fontsize=12)

    plt.hist(df.myt_quantity, bins=np.linspace(4.5, 10.5, 7), rwidth=0.8)
    save_visualisation(filename, img_dir)


def animate_simulation(out_dirs, myt_quantity):
    """

    :param out_dirs: directory containing the simulation runs
    :param myt_quantity: number of agents
    """
    run_states = []
    max_timestep = 0
    target = None

    for out_dir in out_dirs:
        run = utils.utils.load_dataset(out_dir, 'complete-simulation.pkl')
        r = run[['name', 'timestep', 'position', 'goal_position']]
        run_states.append(r)

        target = np.array(r[r['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

        max_ts = r['timestep'].max()
        if max_ts > max_timestep:
            max_timestep = max_ts

    timesteps = np.arange(max_timestep)

    fig = plt.figure(1, figsize=(8.8, 6.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, max_timestep - 1)

    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('x position', fontsize=11)
    plt.yticks(target)
    plt.grid()

    ax1 = ax.twinx()
    thymio_names = []
    for i in range(myt_quantity):
        thymio_names.append('myt%d' % (i + 1))

    ax1.set_yticklabels(thymio_names)
    ax1.yaxis.set_ticks(target)

    ax.set_ylim(
        target[0] - 10,
        target[myt_quantity - 1] + 10
    )
    ax1.set_ylim(
        target[0] - 10,
        target[myt_quantity - 1] + 10
    )

    plt.title('Animations', weight='bold', fontsize=12)

    xs = []
    lines = []
    inits_l = np.full([32, max_timestep], np.nan)

    # colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    # labels = ['omniscient controller', 'manual controller', 'distributed controller', 'distributed controller (* 5)',
    #           'communication controller', 'communication controller (* 5)']

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = ['omniscient controller', 'manual controller', 'distributed controller', 'communication controller']

    for c, controller in enumerate(run_states):
        for i, name in enumerate(controller.name.unique()):
            if i == 0 or i == myt_quantity - 1:
                colour = 'black'
                label = None
                alpha = None
            else:
                colour = colours[c]
                label = labels[c]
                alpha = 0.8

            x = np.array(controller[controller['name'] == name].apply(lambda row: list(row.position)[0], axis=1))
            x = np.pad(x, ((0, len(timesteps) - len(x))), mode='edge')
            if name == 'distributed controller (* 5)' or name == 'communication controller (* 5)':
                line, = ax.plot(timesteps, x, color=colour, label=label, alpha=alpha, ls='--')
            else:
                line, = ax.plot(timesteps, x, color=colour, label=label, alpha=alpha)

            xs.append(x)
            lines.append(line)

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[6], handles[12], handles[18]] #, handles[24], handles[30]] #FIXME
    labels = [labels[0], labels[6], labels[12], labels[18]] #, labels[24], labels[30]]

    lgd = ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

    def init():
        for idx, line in enumerate(lines):
            line.set_ydata(inits_l[idx])

    def animate(i):
        for idx, line in enumerate(lines):
            inits_l[idx][i] = xs[idx][i]
            line.set_ydata(inits_l[idx])  # update the data.

    # create animation using the animate() function
    myAnimation = animation.FuncAnimation(fig, animate, init_func=init, frames=max_timestep)

    video_path = os.path.join(os.path.dirname(out_dirs[0]), 'animation.mp4')
    myAnimation.save(video_path, dpi=300)
    plt.close()


def plot_simulations(out_dirs, myt_quantity):
    """

    :param out_dirs: directory containing the simulation runs
    :param myt_quantity: number of agents
    """
    run_states = []
    max_timestep = 0
    target = None

    for out_dir in out_dirs:
        run = utils.utils.load_dataset(out_dir, 'complete-simulation.pkl')
        r = run[['name', 'timestep', 'position', 'goal_position']]
        run_states.append(r)

        target = np.array(r[r['timestep'] == 1].apply(lambda row: list(row.goal_position)[0], axis=1))

        max_ts = r['timestep'].max()
        if max_ts > max_timestep:
            max_timestep = max_ts

    timesteps = np.arange(max_timestep)

    colours = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = ['omniscient controller', 'manual controller', 'distributed controller', 'communication controller']

    thymio_names = []
    for i in range(myt_quantity):
        thymio_names.append('myt%d' % (i + 1))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(10.8, 6.8), constrained_layout=True, nrows=2, ncols=2)
    # fig, axes = plt.subplots(figsize=(6.8, 10.8), constrained_layout=True, nrows=3)
    axes = [ax1, ax2, ax3, ax4]

    for c, controller in enumerate(run_states):
        axes[c].set_title('%s' % labels[c], weight='bold', fontsize=12)

        for i, name in enumerate(controller.name.unique()):
            if i == 0 or i == myt_quantity - 1:
                colour = 'black'
                label = None
                alpha = None
            else:
                colour = colours[c]
                label = labels[c]
                alpha = 0.8

            x = np.array(controller[controller['name'] == name].apply(lambda row: list(row.position)[0], axis=1))
            x = np.pad(x, ((0, len(timesteps) - len(x))), mode='edge')
            axes[c].plot(timesteps, x, color=colour, label=label, alpha=alpha)
            axes[c].fill_between(timesteps, x - 2.95, x + 7.95, alpha=0.2, color=colour)

    for idx, ax in enumerate(fig.get_axes()):
        if idx == 2 or idx == 3:
            ax.set(xlabel='timestep')
        if idx == 0 or idx == 2:
            ax.set(ylabel='x position')

        ax.set_xlim(0, max_timestep - 1)

        ax.set_ylim(
            target[0] - 10,
            target[myt_quantity - 1] + 10
        )

        if idx == 0:
            ax.tick_params(bottom=True, labelbottom=False)
            ax.set_yticks(target)
            ax.yaxis.tick_left()
            ax.yaxis.set_label_position("left")
        elif idx == 1:
            ax.tick_params(bottom=True, labelbottom=False)
            ax.yaxis.set_ticks(target)
            ax.set_yticklabels(thymio_names)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        elif idx == 2:
            ax.tick_params(bottom=True, labelbottom=True)
            ax.set_yticks(target)
            ax.yaxis.tick_left()
            ax.yaxis.set_label_position("left")
        elif idx == 3:
            ax.tick_params(bottom=True, labelbottom=True)
            ax.yaxis.set_ticks(target)
            ax.set_yticklabels(thymio_names)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        ax.grid()

    save_visualisation('Positions over time', os.path.dirname(out_dirs[0]))


def test_controller_given_init_positions(model_img, model, net_input):
    """

    :param model_img: directory for the output image of the model
    :param model: name of the model
    :param net_input: input of the network (between: prox_values, prox_comm and all_sensors)
    """
    from generate_simulation_data import GenerateSimulationData as g
    from controllers import controllers_task1 as controllers
    import matplotlib.patches as mpatches

    myt_quantity = 3
    min_distance = 10.9

    omniscient_controller_factory = g.get_controller('omniscient', controllers, 'distribute', 3, net_input)

    manual_controller_factory = g.get_controller('manual', controllers, 'distribute', 3, net_input)

    distributed_net_dir = os.path.join('models', 'task1', 'distributed', model)
    distributed_controller_factory = g.get_controller('learned', controllers, 'distribute', 3, net_input,
                                                      model=model, model_dir=distributed_net_dir,
                                                      communication=False)

    communication_net_dir = os.path.join('models', 'task1', 'communication', model)
    communication_controller_factory = g.get_controller('learned', controllers, 'distribute', 3, net_input,
                                                        model=model, model_dir=communication_net_dir,
                                                        communication=True)

    controller_factories = [(omniscient_controller_factory, 'omniscient'),
                            (manual_controller_factory, 'manual'),
                            (distributed_controller_factory, 'distributed'),
                            (distributed_controller_factory, 'distributed (control * 5)'),
                            (communication_controller_factory, 'communication'),
                            (communication_controller_factory, 'communication (control * 5)')]

    controllers_predictions = []
    std_controllers_predictions = []

    simulations = 10 * 10
    max_range = 48
    x = np.linspace(0, max_range, num=simulations)

    for factory, name in controller_factories:
        world, myts = g.setup(factory, myt_quantity)

        control_predictions = []
        std_control_predictions = []

        for simulation in tqdm.tqdm(x):
            control_prediction = []

            for _ in range(100):
                epsilon = np.random.uniform(-0.5, 0.5)
                g.init_positions(myts, net_input, max_range/2, variate_pose=True, x=simulation, epsilon=epsilon)

                world.step(dt=0.1)
                control = myts[1].motor_left_target

                if name == 'distributed (control * 5)' or name == 'communication (control * 5)':
                    control = control * 5

                control = min(max(-16.6, control), 16.6)
                control_prediction.append(control)

            prediction = np.mean(control_prediction)
            std_prediction = np.std(control_prediction)

            control_predictions.append(prediction)
            std_control_predictions.append(std_prediction)

        controllers_predictions.append(control_predictions)
        std_controllers_predictions.append(std_control_predictions)

    title = 'Controllers response by varying init position - %s' % model
    file_name = 'controllers-response-varying_init_position-%s' % model

    # Plot the output of the network
    utils.utils.check_dir(model_img)

    plt.figure()
    plt.xlabel('x position', fontsize=11)
    plt.ylabel('control', fontsize=11)

    for idx, el in enumerate(controllers_predictions):
        y = np.array(el)
        std = np.array(std_controllers_predictions[idx])

        if controller_factories[idx][1] == 'distributed (control * 5)' or controller_factories[idx][1] == 'communication (control * 5)':
            plt.plot(x + min_distance, y, label=controller_factories[idx][1], ls='--')
        else:
            plt.plot(x + min_distance, y, label=controller_factories[idx][1])
        plt.fill_between(x + min_distance,
                         (y - std).clip(-16.6, 16.6),
                         (y + std).clip(-16.6, 16.6),
                         alpha=0.15, label=controller_factories[idx][1])

    plt.title(title, weight='bold', fontsize=12)

    colors = ["w", "w"]
    texts = ["mean", "+/- 1 std"]
    patches = [mpatches.Patch(color=colors[i], label="{:s}".format(texts[i])) for i in range(len(texts))]

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    handles = [patches[0], handles[0], handles[1], handles[2], handles[3], handles[4], handles[5],
               patches[1], handles[6], handles[7], handles[8], handles[9], handles[10], handles[11]]
    labels = [texts[0], labels[0], labels[1], labels[2], labels[3], labels[4], labels[5],
              texts[1], labels[6], labels[7], labels[8], labels[9], labels[10], labels[11]]

    plt.legend(handles=handles, labels=labels, loc='lower center', fontsize=11, bbox_to_anchor=(0.5, -0.7), ncol=2)

    major_xticks = [(max_range + min_distance * 2) / 2]
    minor_xticks = np.linspace(0 + min_distance, max_range + min_distance, 9)
    minor_xticks = np.setdiff1d(minor_xticks, major_xticks)
    major_yticks = [0]
    minor_yticks = np.round(np.linspace(-16.6, 16.6, 7), 2)
    minor_yticks = np.setdiff1d(minor_yticks, major_yticks)

    ax.minorticks_on()

    ax.set_xticks(major_xticks)
    ax.set_yticks(major_yticks)

    ax.set_xticks(minor_xticks, minor=True)
    ax.set_yticks(minor_yticks, minor=True)

    ax.set_xticklabels(minor_xticks, minor=True)
    ax.set_yticklabels(minor_yticks, minor=True)

    ax.tick_params(axis='x', which='both', labelrotation=45, labelright=True)
    ax.tick_params(axis='y', which='both')

    # Specify different settings for major and minor grids
    ax.grid(which='major')

    save_visualisation(file_name, model_img)

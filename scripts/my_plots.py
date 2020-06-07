import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style="white")
from sklearn.linear_model import LinearRegression
import utils


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
    # plt.show()
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

    max_time_step = runs['timestep'].max()
    time_steps = np.arange(max_time_step)

    # position_distances = runs_sub.groupby(['timestep', 'run',]).mean().reset_index()

    v = utils.cartesian_product(runs_sub.timestep.unique(), runs_sub.run.unique(), runs_sub.name.unique())
    idx = pd.MultiIndex.from_arrays([v[:, 0], v[:, 1], v[:, 2]])

    position_distances = runs_sub.set_index(['timestep', 'run', 'name']).reindex(idx)
    position_distances.index.names = ['timestep', 'run', 'name']
    position_distances = position_distances.reset_index()
    position_distances = position_distances.fillna(0).drop(columns='run')

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


def plot_compared_distance_from_goal(runs_dir_omniscient, runs_dir_manual, runs_dir_learned, img_dir, title, filename, net_input):
    """

    :param runs_dir_omniscient:
    :param runs_dir_manual:
    :param runs_dir_learned:
    :param img_dir:
    :param title:
    :param filename:
    :param net_input:
    :return:
    """

    dist_from_goal_o = []
    dist_from_goal_m = []
    dist_from_goal_l = []

    time_steps_o, _, _, _, _, \
    mean_dist_from_goal_o, std_dist_from_goal_o = utils.get_pos_sensing_control(runs_dir_omniscient,
                                                                                net_input,
                                                                                dist_from_goal_o)

    time_steps_m, _, _, _, _, \
    mean_dist_from_goal_m, std_dist_from_goal_m = utils.get_pos_sensing_control(runs_dir_manual,
                                                                                net_input,
                                                                                dist_from_goal_m)

    time_steps_l, _, _, _, _, \
    mean_dist_from_goal_l, std_dist_from_goal_l = utils.get_pos_sensing_control(runs_dir_learned,
                                                                                net_input,
                                                                                dist_from_goal_l)

    plt.figure()
    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('distance from goal', fontsize=11)
    plt.ylim(0, 4)

    mean_dist_o = np.nanmean(mean_dist_from_goal_o, axis=0)
    std_dist_o = np.nanstd(std_dist_from_goal_o, axis=0)
    plt.plot(time_steps_o, mean_dist_o, label='omniscient mean')
    plt.fill_between(time_steps_o, mean_dist_o - std_dist_o, mean_dist_o + std_dist_o, alpha=0.2,
                     label='omniscient +/- 1 std')

    mean_dist_m = np.nanmean(mean_dist_from_goal_m, axis=0)
    std_dist_m = np.nanstd(std_dist_from_goal_m, axis=0)
    plt.plot(time_steps_m, mean_dist_m, label='manual mean')
    plt.fill_between(time_steps_m, mean_dist_m - std_dist_m, mean_dist_m + std_dist_m, alpha=0.2,
                     label='manual +/- 1 std')

    mean_dist_l = np.nanmean(mean_dist_from_goal_l, axis=0)
    std_dist_l = np.nanstd(std_dist_from_goal_l, axis=0)
    plt.plot(time_steps_l, mean_dist_l, label='learned mean')
    plt.fill_between(time_steps_l, mean_dist_l - std_dist_l, mean_dist_l + std_dist_l, alpha=0.2,
                     label='learned +/- 1 std')

    plt.legend()
    plt.title(title, weight='bold', fontsize=12)
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
    target = np.array(run.apply(lambda row: list(row.goal_position)[0], axis=1))

    max_time_step = run['timestep'].max()
    time_steps = np.arange(max_time_step)

    proximity_sensors = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']

    run_myt2 = run[run['name'] == 'myt2'].drop(columns='name').reset_index()
    _, myt2_control, run_myt2 = utils.extract_input_output(run_myt2, net_input)
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

    target = np.array(runs_sub[runs_sub['run'] == 0].apply(lambda row: list(row.goal_position)[0], axis=1))

    proximity_sensors = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']

    runs_myt2 = runs[runs['name'] == 'myt2'].drop(columns='name').reset_index()
    _, _, runs_myt2 = utils.extract_input_output(runs_myt2, net_input)

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

    plt.xticks(x)

    plt.plot(x, train_loss, label='train')
    plt.plot(x, valid_loss, label='validation')
    if scale is not None:
        plt.ylim(0, scale)

    plt.yscale('log')

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
    plt.figure()

    plt.yscale('log')
    plt.xlabel(x_label, fontsize=11)

    plt.title(title, weight='bold', fontsize=12)

    if label is None:
        plt.hist(prediction, bins=50)
    else:
        plt.hist(prediction, label=label)
        plt.legend(loc='lower center', fontsize=11, bbox_to_anchor=(0.5, -0.5), ncol=len(label),
                   title="proximity sensor", title_fontsize=11, markerscale=0.2)
        plt.tight_layout()

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
    plt.plot(x, lr.predict(np.reshape(x, [-1, 1])), color="orange", label='regression: $R^2=%.3f$' % score)

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

    plt.plot(x, y)

    plt.title(title, weight='bold', fontsize=12)

    save_visualisation(filename, img_dir)


def plot_sensing_timestep(runs_dir, img_dir, net_input, model):
    """

    :param x
    :param y
    :param x_label
    :param img_dir
    :param title
    :param filename
    :param index: this parameter is different from None only when x is the input sensing, otherwise, x is a 1D vector
    """
    # FIXME
    time_steps = []
    sensing = []

    pickle_file = os.path.join(runs_dir, 'complete-simulation.pkl')
    runs = pd.read_pickle(pickle_file)

    for run in runs:
        run_time_steps = np.arange(len(run)).tolist()
        utils.extract_input(run, sensing, net_input)

        time_steps.append(run_time_steps)

    length = max(map(len, time_steps))
    time_steps = np.arange(length)

    length2 = len(sensing[0][0])
    length3 = len(sensing[0][0][0])

    for el1 in sensing:
        el1.extend([[[]]] * (length - len(el1)))
        for el2 in el1:
            el2.extend([[]] * (length2 - len(el2)))
            for el3 in el2:
                el3.extend([np.nan] * (length3 - len(el3)))

    sensing = np.array(sensing)

    # Mean of the sensing of each run, among all the robots
    mean_sensing = np.nanmean(np.nanmean(sensing, axis=0), axis=1)
    std_sensing = np.nanstd(np.nanstd(sensing, axis=0), axis=1)

    proximity_sensors = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']
    plt.figure()

    # Plot the evolution of the sensing over time
    plt.ylabel('sensing (%s)' % net_input, fontsize=11)

    for i in range(np.shape(mean_sensing)[1]):
        plt.plot(time_steps, mean_sensing[:, i], label=proximity_sensors[i])  # , color='black')
        plt.fill_between(time_steps,
                         mean_sensing[:, i] - std_sensing[:, i],
                         mean_sensing[:, i] + std_sensing[:, i],
                         alpha=0.2)
    plt.legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.5), ncol=np.shape(mean_sensing)[1],
               title="proximity sensor")

    plt.xlabel('timestep', fontsize=11)
    title = 'Response Sensing %s' % model
    plt.title(title, weight='bold', fontsize=12)

    plt.tight_layout()

    file_name = 'response-sensing-%s' % model
    save_visualisation(file_name, img_dir)

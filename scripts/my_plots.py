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
    plt.savefig(img)
    # plt.show()
    plt.close()


def plot_distance_from_goal(runs_dir, img_dir, title, filename, net_input):
    """
    :param runs_dir:
    :param img_dir:
    :param title
    :param filename
    :param net_input
    """
    distance_from_goal = []

    time_steps, _, _, _, _, \
    mean_distance_from_goal, std_distance_from_goal = utils.get_pos_sensing_control(runs_dir,
                                                                                    net_input,
                                                                                    distance_from_goal)

    plt.figure()
    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('distance from goal', fontsize=11)
    plt.ylim(0, 4)

    mean_distances = np.nanmean(mean_distance_from_goal, axis=0)
    std_distances = np.nanstd(std_distance_from_goal, axis=0)
    plt.plot(time_steps, mean_distances, label='mean')
    plt.fill_between(time_steps, mean_distances - std_distances, mean_distances + std_distances, alpha=0.2,
                     label='+/- 1 std')

    plt.legend()

    plt.title(title, weight='bold', fontsize=12)
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
    x_positions = []
    myt2_sensing = []
    myt2_control = []

    pickle_file = os.path.join(runs_dir, 'complete-simulation.pkl')
    runs = pd.read_pickle(pickle_file)

    run = runs[simulation]

    time_steps = np.arange(len(run))
    target = utils.extract_run_data(myt2_control, myt2_sensing, run, time_steps, x_positions, net_input)

    x_positions = np.array(x_positions[0])
    myt2_sensing = np.array(myt2_sensing[0])
    myt2_control = np.array(myt2_control[0])

    proximity_sensors = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']

    plt.figure()
    fig, axes = plt.subplots(nrows=3, figsize=(7, 11), sharex=True)

    # Plot the evolution of the positions of all robots over time
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    for i in range(np.shape(x_positions)[1]):
        axes[0].plot(time_steps, x_positions[:, i], label='myt%d' % (i + 1))  # , color='black')
    axes[0].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23),
                   ncol=np.shape(x_positions)[1], title="robot")
    axes[0].set_yticks(target)
    axes[0].grid()

    # Plot, for a given robot, the evolution of sensing over time
    axes[1].set_ylabel('sensing', fontsize=11)
    axes[1].set_title('Thymio 2 Sensing', weight='bold', fontsize=12)
    for i in range(np.shape(myt2_sensing)[1]):
        axes[1].plot(time_steps, myt2_sensing[:, i], label=proximity_sensors[i])
    axes[1].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23),
                   ncol=np.shape(myt2_sensing)[1], title="proximity sensor")
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


def visualise_simulations_comparison_seaborn(img_dir, myt2_control, myt2_sensing, proximity_sensors, target, time_steps,
                                             title, x_positions):
    """

    :param img_dir:
    :param myt2_control:
    :param myt2_sensing:
    :param proximity_sensors:
    :param target:
    :param time_steps:
    :param title:
    :param x_positions:
    :return:
    """

    # Seaborn visualisation
    df_x_positions, df_sensing, df_control = utils.extract_flatten_dataframe(myt2_control, myt2_sensing,
                                                                             time_steps, x_positions)

    plt.figure()
    fig, axes = plt.subplots(nrows=3, figsize=(7, 11), sharex=True)
    labels = []

    for i in range(df_x_positions.shape[1] - 1):
        sns.lineplot(x="timestep", y="myt%d_x_positions" % (i + 2), data=df_x_positions, ax=axes[0])
        labels.append('myt%d' % (i + 1))
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    axes[0].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23), labels=tuple(labels),
                   ncol=df_x_positions.shape[1] - 1, title="robot")
    axes[0].set_yticks(target)
    axes[0].grid()
    axes[0].set(ylabel='x position')

    for i in range(df_sensing.shape[1] - 1):
        sns.lineplot(x="timestep", y="myt2_sensor_%d" % (i + 1), data=df_sensing, ax=axes[1])
    axes[1].set_title('Thymio 2 Sensing', weight='bold', fontsize=12)
    axes[1].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23), labels=tuple(proximity_sensors),
                   ncol=(df_sensing.shape[1] - 1), title="proximity sensor")
    axes[1].grid()
    axes[1].set(ylabel='sensing')

    sns.lineplot(x="timestep", y="myt2_control", data=df_control, ax=axes[2])
    axes[2].set_title('Thymio 2 Control', weight='bold', fontsize=12)
    axes[2].grid()
    axes[2].set(ylabel='control')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    plt.xlabel('timestep', fontsize=11)
    fig.suptitle(title, fontsize=14, weight='bold')

    filename = 'compare-simulation-seaborn'
    save_visualisation(filename, img_dir, make_space=True, axes=axes)


def visualise_simulations_comparison(runs_dir, img_dir, title, net_input, seaborn=False):
    """
    :param runs_dir:
    :param img_dir:
    :param title
    :param seaborn
    """
    time_steps, x_positions, myt2_sensing, myt2_control, target, _, _ = utils.get_pos_sensing_control(runs_dir, net_input)

    mean_x_positions = np.nanmean(x_positions, axis=0)
    mean_myt2_control = np.nanmean(myt2_control, axis=0)
    mean_myt2_sensing = np.nanmean(myt2_sensing, axis=0)

    std_x_positions = np.nanstd(x_positions, axis=0)
    std_myt2_control = np.nanstd(myt2_control, axis=0)
    std_myt2_sensing = np.nanstd(myt2_sensing, axis=0)

    proximity_sensors = ['fll', 'fl', 'fc', 'fr', 'frr', 'bl', 'br']

    plt.figure()
    fig, axes = plt.subplots(nrows=3, figsize=(7, 11), sharex=True)

    # Plot the evolution of the positions of all robots over time
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio positions over time', weight='bold', fontsize=12)
    for i in range(np.shape(mean_x_positions)[1]):
        axes[0].plot(time_steps, mean_x_positions[:, i], label='myt%d' % (i + 1))
        axes[0].fill_between(time_steps,
                             mean_x_positions[:, i] - std_x_positions[:, i],
                             mean_x_positions[:, i] + std_x_positions[:, i],
                             alpha=0.2)
    axes[0].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23),
                   ncol=np.shape(mean_x_positions)[1], title="robot")
    axes[0].set_yticks(target)
    axes[0].grid()

    # Plot, for a given robot, the evolution of sensing over time
    axes[1].set_ylabel('sensing', fontsize=11)
    axes[1].set_title('Thymio 2 Sensing', weight='bold', fontsize=12)
    for i in range(np.shape(mean_myt2_sensing)[1]):
        axes[1].plot(time_steps, mean_myt2_sensing[:, i], label=proximity_sensors[i])  # , color='black')
        axes[1].fill_between(time_steps,
                             mean_myt2_sensing[:, i] - std_myt2_sensing[:, i],
                             mean_myt2_sensing[:, i] + std_myt2_sensing[:, i],
                             alpha=0.2)
    axes[1].legend(loc='lower center', fontsize='small', bbox_to_anchor=(0.5, -0.23),
                   ncol=np.shape(mean_myt2_sensing)[1], title="proximity sensor")
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

    plt.xlabel('timestep', fontsize=11)
    fig.suptitle(title, fontsize=14, weight='bold')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    filename = 'compare-simulation'
    save_visualisation(filename, img_dir, make_space=True, axes=axes)

    if seaborn:
        visualise_simulations_comparison_seaborn(img_dir, myt2_control, myt2_sensing, proximity_sensors, target,
                                                 time_steps, title, x_positions)


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

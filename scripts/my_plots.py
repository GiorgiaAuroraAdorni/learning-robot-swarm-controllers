import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from utils import extract_run_data, check_dir, get_pos_sensing_control, extract_flatten_dataframe

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
    plt.savefig(img)
    # plt.show()
    plt.close()


def plot_distance_from_goal(runs_dir, img_dir, title, filename):
    """
    :param runs_dir:
    :param img_dir:
    :param title
    :param seaborn
    """
    distance_from_goal = []
    time_steps, _, _, _, _, distance_from_goal = get_pos_sensing_control(runs_dir, distance_from_goal)

    plt.figure()
    plt.xlabel('timestep', fontsize=11)
    plt.ylabel('mean distance from goal', fontsize=11)
    plt.ylim(0, 4)

    plt.plot(time_steps, np.nanmean(distance_from_goal, axis=0))

    plt.title(title, weight='bold', fontsize=12)
    save_visualisation(filename, img_dir)


def visualise_simulation(runs_dir, img_dir, simulation, title):
    """

    :param runs_dir:
    :param img_dir:
    :param simulation:
    :param title:
    """
    run = None

    x_positions = []
    myt2_sensing = []
    myt2_control = []

    for file_name in os.listdir(runs_dir):
        if not file_name.endswith('-%d.pkl' % simulation) or not file_name.startswith('complete'):
            continue

        pickle_file = os.path.join(runs_dir, file_name)
        run = pd.read_pickle(pickle_file)

    time_steps = np.arange(len(run))
    target = extract_run_data(myt2_control, myt2_sensing, run, time_steps, x_positions)

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

    # plot distance from goal
    plt.figure()


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
    df_x_positions, df_sensing, df_control = extract_flatten_dataframe(myt2_control, myt2_sensing, time_steps,
                                                                       x_positions)

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


def visualise_simulations_comparison(runs_dir, img_dir, title, seaborn=False):
    """
    :param runs_dir:
    :param img_dir:
    :param title
    :param seaborn
    """
    time_steps, x_positions, myt2_sensing, myt2_control, target = get_pos_sensing_control(runs_dir)

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

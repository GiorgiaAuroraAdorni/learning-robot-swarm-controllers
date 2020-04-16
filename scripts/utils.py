import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

    for file_name in os.listdir(runs_dir):
        if not file_name.endswith('-0.pkl'):
            continue

        pickle_file = os.path.join(runs_dir, file_name)
        run = pd.read_pickle(pickle_file)

    time_steps = np.arange(len(run))
    x_positions = []
    myt2_sensing = []
    myt2_control = []

    for step in run:
        x_pos = []
        for myt in step:
            # The input is the prox_values, that are the response values of ​​the sensors [array of 7 floats]
            x_position = myt['position'][0]
            x_pos.append(x_position)

            if myt['name'] == 'myt2':
                sensing = myt['prox_values'].copy()
                control = myt['motor_left_target']

                myt2_sensing.append(sensing)
                myt2_control.append(control)

        x_positions.append(x_pos)

    x_positions = np.array(x_positions)
    myt2_sensing = np.array(myt2_sensing)
    myt2_control = np.array(myt2_control)

    grid = np.linspace(0, x_positions[-1, -1], 5)
    fig, axes = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)

    # Plot the evolution of the positions of all robots over time
    # axes[0].set_xlabel('timestep', fontsize=11)
    axes[0].set_ylabel('x position', fontsize=11)
    axes[0].set_title('Thymio position over time', weight='bold', fontsize=12)
    for i in range(np.shape(x_positions)[1]):
        axes[0].plot(time_steps, x_positions[:, i], label='myt%d' % (i + 1))  #, color='black')
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


def plot_train_test_losses(train_loss, test_loss, img_dir, title, filename, scale=None):
    """

    :param train_loss: the training losses
    :param test_loss: the testing losses
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
    # plt.plot(x, valid_loss, label='Valid ' + title)
    plt.plot(x, test_loss, label='Test ' + title)
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
img_dir = 'models/distributed/images/'

visualise_simulation(runs_dir, img_dir, model)

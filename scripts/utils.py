import os
import numpy as np
import matplotlib.pyplot as plt


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


def scatter_plot(x, y, x_label, y_label, img_dir, title, filename):
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


def plot_prediction_histogram(prediction, x_label, img_dir, title, filename, label=None):
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

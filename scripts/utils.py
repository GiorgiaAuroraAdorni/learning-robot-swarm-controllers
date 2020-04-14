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


def plot_setting(train_loss, test_loss, out_dir, model):
    """

    :param valid_accuracies: OrderedDict containing the model and the correspondent validation accuracy
    :param out_dir: project directory for the output files
    :param model_dir: model directory
    :param epoch: number of epochs of the model
    :param final: boolean parameter that is True if only if the model is the one selected for the final test
    """
    img_dir = out_dir + 'img/'
    check_dir(img_dir)

    my_plot(train_loss, test_loss, model, img_dir)


def my_plot(train_loss, test_loss, model, img_dir):
    """

    :param train_loss: the training losses
    :param test_loss: the testing losses
    :param model: name of the model
    :param img_dir: directory for the output image
    :param title: title of the plot
    """
    title = 'Loss %s' % model

    x = np.arange(1, len(train_loss) + 1, dtype=int)

    plt.xlabel('epoch', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    plt.plot(x, train_loss, label='Train ' + title)
    # plt.plot(x, valid_loss, label='Valid ' + title)
    plt.plot(x, test_loss, label='Test ' + title)
    # plt.ylim(0, 4.5)

    plt.legend()
    plt.title(title, weight='bold', fontsize=12)

    file = os.path.join(img_dir, '%s.png' % title)
    plt.savefig(file)
    plt.show()

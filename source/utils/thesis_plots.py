import os

from my_plots import save_visualisation
from utils import check_dir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_losses_compressed(model_img_dir, models, task, type, name):
    """

    :param model_img_dir:
    :param models:
    :param task:
    :param type:
    :param name
    """

    models_dir = []
    for el in models:
        model_dir = os.path.join('models', task, type, el)
        models_dir.append(model_dir)

    fig = plt.figure(constrained_layout=True, figsize=(9, 6))
    plt.xlabel('epoch', fontsize=11)
    plt.ylabel('loss', fontsize=11)

    for idx, el in enumerate(models):

        file_losses = os.path.join(models_dir[idx], 'losses.npy')

        losses = pd.read_pickle(file_losses)
        train_loss = losses.loc[:, 't. loss']
        valid_loss = losses.loc[:, 'v. loss']

        x = np.arange(len(train_loss), dtype=int)

        ln, = plt.plot(x, valid_loss, label='%s validation loss' % el)
        plt.plot(x, train_loss, label='%s train loss' % el, color=ln.get_color(), ls='--', alpha=0.5)

    filename = 'loss-%s-%s' % (type, name)

    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles=handles, labels=labels, loc='upper left', fontsize='small',
               bbox_to_anchor=(1.05, 1))

    if task == 'task2':
        plt.ylim(0, 0.4)
        plt.xlim(0, 100)

    save_visualisation(filename, model_img_dir)


# # # #

task = 'task1'
type = 'distributed'
subtype = 'distributed'
model_img_dir = os.path.join('models', task, type, subtype)
check_dir(model_img_dir)

name = 'all'
models = ['net-d1', 'net-d2', 'net-d3', 'net-d4', 'net-d5', 'net-d6', 'net-d7', 'net-d8', 'net-d9']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_8'
models = ['net-d1', 'net-d4', 'net-d7']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_13'
models = ['net-d2', 'net-d5', 'net-d8']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_24'
models = ['net-d3', 'net-d6', 'net-d9']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'prox_values'
models = ['net-d1', 'net-d2', 'net-d3']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'prox_comm'
models = ['net-d4', 'net-d5', 'net-d6']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'all_sensors'
models = ['net-d7', 'net-d8', 'net-d9']
plot_losses_compressed(model_img_dir, models, task, type, name)

# # # #

subtype = 'distributed-extension'
model_img_dir = os.path.join('models', task, type, subtype)
check_dir(model_img_dir)

name = 'all'
models = ['net-d10', 'net-d11', 'net-d12', 'net-d13', 'net-d14', 'net-d15', 'net-d16', 'net-d17', 'net-d18']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_8'
models = ['net-d1', 'net-d13', 'net-d16']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_20'
models = ['net-d11', 'net-d14', 'net-d17']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_var'
models = ['net-d12', 'net-d15', 'net-d18']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'N5'
models = ['net-d10', 'net-d11', 'net-d12']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'N8'
models = ['net-d13', 'net-d14', 'net-d15']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'Nvar'
models = ['net-d16', 'net-d17', 'net-d18']
plot_losses_compressed(model_img_dir, models, task, type, name)

# # # #

type = 'communication'
subtype = 'communication-extension'
model_img_dir = os.path.join('models', task, type, subtype)
check_dir(model_img_dir)

name = 'all'
models = ['net-c1', 'net-c2', 'net-c3', 'net-c4', 'net-c5', 'net-c6', 'net-c7', 'net-c8', 'net-c9']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_8'
models = ['net-c1', 'net-c3', 'net-c7']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_13'
models = ['net-c2', 'net-c5', 'net-c8']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_24'
models = ['net-c3', 'net-c6', 'net-c9']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'prox_values'
models = ['net-c1', 'net-c2', 'net-c3']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'prox_comm'
models = ['net-c4', 'net-c5', 'net-c6']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'all_sensors'
models = ['net-c7', 'net-c8', 'net-c9']
plot_losses_compressed(model_img_dir, models, task, type, name)

# # # #

subtype = 'communication-extension'
model_img_dir = os.path.join('models', task, type, subtype)
check_dir(model_img_dir)

name = 'all'
models = ['net-c10', 'net-c11', 'net-c12', 'net-c13', 'net-c14', 'net-c15', 'net-c16', 'net-c17', 'net-c18']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_8'
models = ['net-c10', 'net-c13', 'net-c16']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_20'
models = ['net-c11', 'net-c14', 'net-c17']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'gap_var'
models = ['net-c12', 'net-c15', 'net-c18']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'N5'
models = ['net-c10', 'net-c11', 'net-c12']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'N8'
models = ['net-c13', 'net-c14', 'net-c15']
plot_losses_compressed(model_img_dir, models, task, type, name)

name = 'Nvar'
models = ['net-c16', 'net-c17', 'net-c18']
plot_losses_compressed(model_img_dir, models, task, type, name)

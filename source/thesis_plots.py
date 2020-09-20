import os

from utils.my_plots import save_visualisation
from utils.utils import check_dir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_losses_compressed(model_img_dir, models, task, type):
    """

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

    filename = 'loss-%s' % type

    ax = fig.gca()
    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles=handles, labels=labels, loc='upper left', fontsize='small',
               bbox_to_anchor=(1.05, 1))

    if task == 'task2':
        plt.ylim(0, 0.4)
        plt.xlim(0, 100)

    save_visualisation(filename, model_img_dir)

task = 'task1'
type = 'distributed'
model_img_dir = os.path.join('models', task, type, 'images')
check_dir(model_img_dir)
models = ['net1', 'net2', 'net3', 'net4', 'net5', 'net6', 'net7', 'net8', 'net9']

plot_losses_compressed(model_img_dir, models, task, type)


task = 'task1'
type = 'communication'
model_img_dir = os.path.join('models', task, type, 'images')
check_dir(model_img_dir)
models = ['net1', 'net2', 'net3', 'net4', 'net5', 'net6', 'net7', 'net8', 'net9']

plot_losses_compressed(model_img_dir, models, task, type)


task = 'task2'
type = 'communication'
model_img_dir = os.path.join('models', task, type, 'images')
check_dir(model_img_dir)
models = ['net-v1', 'net-v2', 'net-v3', 'net-v4', 'net-v5', 'net-v6', 'net-v7', 'net-v8', 'net-v9']

plot_losses_compressed(model_img_dir, models, task, type)


task = 'task1'
type = 'distributed'
model_img_dir = os.path.join('models', task, type, 'images-avg_gap8')
check_dir(model_img_dir)
models = ['net1', 'net4', 'net7']

plot_losses_compressed(model_img_dir, models, task, type)

task = 'task1'
type = 'distributed'
model_img_dir = os.path.join('models', task, type, 'images-avg_gap13')
check_dir(model_img_dir)
models = ['net2', 'net5', 'net8']

plot_losses_compressed(model_img_dir, models, task, type)


task = 'task1'
type = 'distributed'
model_img_dir = os.path.join('models', task, type, 'images-avg_gap24')
check_dir(model_img_dir)
models = ['net3', 'net6', 'net9']

plot_losses_compressed(model_img_dir, models, task, type)


task = 'task1'
type = 'distributed'
model_img_dir = os.path.join('models', task, type, 'images-prox_values')
check_dir(model_img_dir)
models = ['net1', 'net2', 'net3']

plot_losses_compressed(model_img_dir, models, task, type)

task = 'task1'
type = 'distributed'
model_img_dir = os.path.join('models', task, type, 'images-prox_comm')
check_dir(model_img_dir)
models = ['net4', 'net5', 'net6']

plot_losses_compressed(model_img_dir, models, task, type)


task = 'task1'
type = 'distributed'
model_img_dir = os.path.join('models', task, type, 'images-all_sensors')
check_dir(model_img_dir)
models = ['net7', 'net8', 'net9']

plot_losses_compressed(model_img_dir, models, task, type)

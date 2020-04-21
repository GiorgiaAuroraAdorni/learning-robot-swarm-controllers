import os

import torch

from controllers import distributed_controllers
from generate_simulation_data import generate_simulation, check_dataset_conformity
from utils import check_dir
from my_plots import visualise_simulation, visualise_simulations_comparison

if __name__ == '__main__':
    MANUAL_CONTROLLER = "manual-controller"
    OMNISCIENT_CONTROLLER = "omniscient-controller"
    LEARNED_CONTROLLER = "learned-controller"

    # TODO define this with argparser
    controller = LEARNED_CONTROLLER

    # TODO same
    model = 'net1'
    # TODO same
    myt_quantity = 5

    dataset = '%dmyts-%s' % (myt_quantity, controller)

    runs_dir = os.path.join('datasets/', dataset)
    img_dir = '%s/images/' % runs_dir

    check_dir(runs_dir)
    check_dir(img_dir)

    if controller == MANUAL_CONTROLLER:
        controller_factory = distributed_controllers.ManualController
    elif controller == OMNISCIENT_CONTROLLER:
        controller_factory = distributed_controllers.OmniscientController
    elif controller == LEARNED_CONTROLLER:
        model_dir = 'models/distributed/%s' % model
        net = torch.load('%s/%s' % (model_dir, model))
        # controller_factory = lambda **kwargs: d_c.LearnedController(net=net, **kwargs)

        def controller_factory(**kwargs):
            return distributed_controllers.LearnedController(net=net, **kwargs)
    else:
        raise ValueError("Invalid value for controller")

    generate_simulation(runs_dir, simulations=1000, controller_factory=controller_factory, myt_quantity=myt_quantity)

    visualise_simulation(runs_dir, img_dir, 0, 'Distribution simulation %d - %s' % (0, controller))
    # visualise_simulation(runs_dir, img_dir, 1, 'Distribution simulation %d - %s' % (1, controller))
    # visualise_simulation(runs_dir, img_dir, 2, 'Distribution simulation %d - %s' % (2, controller))
    # visualise_simulation(runs_dir, img_dir, 3, 'Distribution simulation %d - %s' % (3, controller))
    # visualise_simulation(runs_dir, img_dir, 4, 'Distribution simulation %d - %s' % (4, controller))
    visualise_simulations_comparison(runs_dir, img_dir, 'Distribution of all simulations - %s' % controller)

    check_dataset_conformity(runs_dir, img_dir, dataset)

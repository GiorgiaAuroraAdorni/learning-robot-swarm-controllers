import os

from generate_simulation_data import generate__simulation
from utils import check_dir, visualise_simulation, visualise_simulations_comparison

if __name__ == '__main__':
    myt_quantity = 5

    model = 'net1'
    controller = 'omniscient'  # model  # 'distributed'  # 'omniscient'

    dataset = '%dmyts-%s' % (myt_quantity, controller)

    runs_dir = os.path.join('datasets/', dataset)
    # model_dir = 'models/distributed/%s' % model

    check_dir(runs_dir)
    # check_dir(model_dir)

    # generate__simulation(runs_dir, simulations=1000, controller=controller, myt_quantity=myt_quantity)
    generate__simulation(runs_dir, simulations=1000, controller=controller, myt_quantity=myt_quantity, model_dir=None)
    img_dir = '%s/images/' % runs_dir

    visualise_simulation(runs_dir, img_dir, 0, 'Distribution simulation %d - %s controller' % (0, controller))
    visualise_simulation(runs_dir, img_dir, 1, 'Distribution simulation %d - %s controller' % (1, controller))
    visualise_simulation(runs_dir, img_dir, 2, 'Distribution simulation %d - %s controller' % (2, controller))
    visualise_simulation(runs_dir, img_dir, 3, 'Distribution simulation %d - %s controller' % (3, controller))
    visualise_simulation(runs_dir, img_dir, 4, 'Distribution simulation %d - %s controller' % (4, controller))
    visualise_simulations_comparison(runs_dir, img_dir, 'Distribution of all simulations - %s controller' % controller)

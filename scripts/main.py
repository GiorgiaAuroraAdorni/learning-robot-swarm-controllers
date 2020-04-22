import os

from distributed import run_distributed
from utils import check_dir
from my_plots import visualise_simulation, visualise_simulations_comparison
from generate_simulation_data import GenerateSimulationData as g


if __name__ == '__main__':
    # TODO define these with argparser
    myt_quantity = 5
    omniscient_controller = "omniscient-controller"

    dataset_omniscient = '%dmyts-%s' % (myt_quantity, omniscient_controller)

    runs_dir_omniscient = os.path.join('datasets/', dataset_omniscient)
    img_dir_omniscient = '%s/images/' % runs_dir_omniscient

    check_dir(runs_dir_omniscient)
    check_dir(img_dir_omniscient)

    print('Generating simulations for %s…' % omniscient_controller)
    # g.generate_simulation(runs_dir_omniscient, simulations=1000, controller=omniscient_controller, myt_quantity=myt_quantity)
    #
    # print('\nGenerating plots for %s…' % omniscient_controller)
    # visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 0,
    #                      'Distribution simulation %d - %s' % (0, omniscient_controller))
    # visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 1,
    #                      'Distribution simulation %d - %s' % (1, omniscient_controller))
    # visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 2,
    #                      'Distribution simulation %d - %s' % (2, omniscient_controller))
    # visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 3,
    #                      'Distribution simulation %d - %s' % (3, omniscient_controller))
    # visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 4,
    #                      'Distribution simulation %d - %s' % (4, omniscient_controller))
    # visualise_simulations_comparison(runs_dir_omniscient, img_dir_omniscient,
    #                                  'Distribution of all simulations - %s' % omniscient_controller)
    #
    # g.check_dataset_conformity(runs_dir_omniscient, img_dir_omniscient, omniscient_controller)
    #
    # # # # #
    manual_controller = "manual-controller"

    dataset_manual = '%dmyts-%s' % (myt_quantity, manual_controller)

    runs_dir_manual = os.path.join('datasets/', dataset_manual)
    img_dir_manual = '%s/images/' % runs_dir_manual

    check_dir(runs_dir_manual)
    check_dir(img_dir_manual)

    # print('\nGenerating simulations for %s…' % manual_controller)
    # g.generate_simulation(runs_dir_manual, simulations=1000, controller=manual_controller, myt_quantity=myt_quantity)
    #
    # print('\nGenerating plots for %s…' % manual_controller)
    # visualise_simulation(runs_dir_manual, img_dir_manual, 0, 'Distribution simulation %d - %s' % (0, manual_controller))
    # visualise_simulation(runs_dir_manual, img_dir_manual, 1, 'Distribution simulation %d - %s' % (1, manual_controller))
    # visualise_simulation(runs_dir_manual, img_dir_manual, 2, 'Distribution simulation %d - %s' % (2, manual_controller))
    # visualise_simulation(runs_dir_manual, img_dir_manual, 3, 'Distribution simulation %d - %s' % (3, manual_controller))
    # visualise_simulation(runs_dir_manual, img_dir_manual, 4, 'Distribution simulation %d - %s' % (4, manual_controller))
    # visualise_simulations_comparison(runs_dir_manual, img_dir_manual, 'Distribution of all simulations - %s' %
    #                                  manual_controller)
    #
    # g.check_dataset_conformity(runs_dir_manual, img_dir_manual, dataset_manual)

    # # # # #
    learned_controller = "learned-controller"

    # TODO define this with argparser
    model = 'net1'
    model_dir = 'models/distributed/%s' % model
    model_img = '%s/images/' % model_dir

    check_dir(model_dir)
    check_dir(model_img)

    file = os.path.join('models/distributed/', 'dataset_split.npy')

    train = False
    if train:
        print('\nTraining %s and generating plots…' % model)
    else:
        print('\nGenerating plots for %s…' % model)
    run_distributed(file, runs_dir_omniscient, model_dir, model_img, model, dataset_omniscient, dataset_manual, train=train)
    #
    # # # # #
    # dataset_learned = '%dmyts-%s' % (myt_quantity, learned_controller)
    #
    # runs_dir_learned = os.path.join('datasets/', dataset_learned)
    # img_dir_learned = '%s/images/' % runs_dir_learned
    #
    # check_dir(runs_dir_learned)
    # check_dir(img_dir_learned)
    #
    # print('\nGenerating simulations for %s…' % learned_controller)
    # g.generate_simulation(runs_dir_learned, simulations=1000, controller=manual_controller, myt_quantity=myt_quantity)
    #
    # print('\nGenerating plots for %s…' % learned_controller)
    # visualise_simulation(runs_dir_learned, img_dir_learned, 0, 'Distribution simulation %d - %s' % (0, manual_controller))
    # visualise_simulation(runs_dir_learned, img_dir_learned, 1, 'Distribution simulation %d - %s' % (1, manual_controller))
    # visualise_simulation(runs_dir_learned, img_dir_learned, 2, 'Distribution simulation %d - %s' % (2, manual_controller))
    # visualise_simulation(runs_dir_learned, img_dir_learned, 3, 'Distribution simulation %d - %s' % (3, manual_controller))
    # visualise_simulation(runs_dir_learned, img_dir_learned, 4, 'Distribution simulation %d - %s' % (4, manual_controller))
    # visualise_simulations_comparison(runs_dir_learned, img_dir_learned, 'Distribution of all simulations - %s' %
    #                                  manual_controller)
    #
    # g.check_dataset_conformity(runs_dir_learned, img_dir_learned, dataset_manual)

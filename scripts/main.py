import argparse
import os

from distributed import run_distributed
from utils import check_dir
from my_plots import visualise_simulation, visualise_simulations_comparison, plot_distance_from_goal
from generate_simulation_data import GenerateSimulationData as g


def Parse():
    """

    :return args
    """
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')

    parser.add_argument('--gui', type=bool, default=False, metavar='N',
                        help='run simulation using the gui (default: False)')
    parser.add_argument('--myt-quantity', type=int, default=5, metavar='N',
                        help='number of thymios for the simulation (default: 5)')
    parser.add_argument('--train-model', type=bool, default=False,
                        help='train the model  (default: False)')
    parser.add_argument('--simulations', type=int, default=1000, metavar='N',
                        help='number of runs for each simulation (default: 1000)')
    parser.add_argument('--model', default='net1', type=str,
                        help='name of the model (default: net1)')
    parser.add_argument('--generate-split', type=bool, default=False, metavar='N',
                        help='generate the indices for the split of the dataset (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = Parse()

    myt_quantity = args.myt_quantity
    omniscient_controller = "omniscient-controller"

    dataset_omniscient = '%dmyts-%s' % (myt_quantity, omniscient_controller)

    runs_dir_omniscient = os.path.join('datasets/', dataset_omniscient)
    img_dir_omniscient = '%s/images/' % runs_dir_omniscient

    check_dir(runs_dir_omniscient)
    check_dir(img_dir_omniscient)

    print('Generating simulations for %s…' % omniscient_controller)
    g.generate_simulation(runs_dir_omniscient, simulations=args.simulations, controller=omniscient_controller,
                          myt_quantity=myt_quantity, args=args)

    print('\nGenerating plots for %s…' % omniscient_controller)
    visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 0,
                         'Distribution simulation %d - %s' % (0, omniscient_controller))
    visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 1,
                         'Distribution simulation %d - %s' % (1, omniscient_controller))
    visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 2,
                         'Distribution simulation %d - %s' % (2, omniscient_controller))
    visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 3,
                         'Distribution simulation %d - %s' % (3, omniscient_controller))
    visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 4,
                         'Distribution simulation %d - %s' % (4, omniscient_controller))
    visualise_simulations_comparison(runs_dir_omniscient, img_dir_omniscient,
                                     'Distribution of all simulations - %s' % omniscient_controller)

    plot_distance_from_goal(runs_dir_omniscient, img_dir_omniscient, 'Robot distance from goal - %s' %
                            omniscient_controller, 'distances-from-goal-%s' % omniscient_controller)
    g.check_dataset_conformity(runs_dir_omniscient, img_dir_omniscient, omniscient_controller)

    # # # # #
    manual_controller = "manual-controller"

    dataset_manual = '%dmyts-%s' % (myt_quantity, manual_controller)

    runs_dir_manual = os.path.join('datasets/', dataset_manual)
    img_dir_manual = '%s/images/' % runs_dir_manual

    check_dir(runs_dir_manual)
    check_dir(img_dir_manual)

    print('\nGenerating simulations for %s…' % manual_controller)
    g.generate_simulation(runs_dir_manual, simulations=args.simulations, controller=manual_controller,
                          myt_quantity=myt_quantity, args=args)

    print('\nGenerating plots for %s…' % manual_controller)
    visualise_simulation(runs_dir_manual, img_dir_manual, 0, 'Distribution simulation %d - %s' % (0, manual_controller))
    visualise_simulation(runs_dir_manual, img_dir_manual, 1, 'Distribution simulation %d - %s' % (1, manual_controller))
    visualise_simulation(runs_dir_manual, img_dir_manual, 2, 'Distribution simulation %d - %s' % (2, manual_controller))
    visualise_simulation(runs_dir_manual, img_dir_manual, 3, 'Distribution simulation %d - %s' % (3, manual_controller))
    visualise_simulation(runs_dir_manual, img_dir_manual, 4, 'Distribution simulation %d - %s' % (4, manual_controller))
    visualise_simulations_comparison(runs_dir_manual, img_dir_manual, 'Distribution of all simulations - %s' %
                                     manual_controller)

    plot_distance_from_goal(runs_dir_manual, img_dir_manual, 'Robot distance from goal - %s' %
                            manual_controller, 'distances-from-goal-%s' % manual_controller)
    g.check_dataset_conformity(runs_dir_manual, img_dir_manual, dataset_manual)

    # # # # #
    learned_controller = "learned-controller"

    model_dir = 'models/distributed/%s' % args.model
    model_img = '%s/images/' % model_dir

    check_dir(model_dir)
    check_dir(model_img)

    file = os.path.join('models/distributed/', 'dataset_split.npy')

    if args.train_model:
        print('\nTraining %s and generating plots…' % args.model)
    else:
        print('\nGenerating plots for %s…' % args.model)
    run_distributed(file, runs_dir_omniscient, model_dir, model_img, args.model, dataset_omniscient, dataset_manual,
                    train=args.train_model, generate_split=args.generate_split)

    # # # # #
    dataset_learned = '%dmyts-%s' % (myt_quantity, learned_controller)

    runs_dir_learned = os.path.join('datasets/', dataset_learned)
    img_dir_learned = '%s/images/' % runs_dir_learned

    check_dir(runs_dir_learned)
    check_dir(img_dir_learned)

    print('\nGenerating simulations for %s…' % learned_controller)
    g.generate_simulation(runs_dir_learned, simulations=args.simulations, controller=learned_controller,
                          myt_quantity=myt_quantity, args=args, model_dir=model_dir, model=args.model)
    #
    print('\nGenerating plots for %s…' % learned_controller)
    visualise_simulation(runs_dir_learned, img_dir_learned, 0, 'Distribution simulation %d - %s' % (0, learned_controller))
    visualise_simulation(runs_dir_learned, img_dir_learned, 1, 'Distribution simulation %d - %s' % (1, learned_controller))
    visualise_simulation(runs_dir_learned, img_dir_learned, 2, 'Distribution simulation %d - %s' % (2, learned_controller))
    visualise_simulation(runs_dir_learned, img_dir_learned, 3, 'Distribution simulation %d - %s' % (3, learned_controller))
    visualise_simulation(runs_dir_learned, img_dir_learned, 4, 'Distribution simulation %d - %s' % (4, learned_controller))
    visualise_simulations_comparison(runs_dir_learned, img_dir_learned, 'Distribution of all simulations - %s' %
                                     learned_controller)

    plot_distance_from_goal(runs_dir_learned, img_dir_learned, 'Robot distance from goal - %s' %
                            learned_controller, 'distances-from-goal-%s' % learned_controller)
    g.check_dataset_conformity(runs_dir_learned, img_dir_learned, dataset_learned)

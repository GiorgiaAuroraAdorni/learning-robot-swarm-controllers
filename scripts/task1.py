import argparse
import os

from distributed import run_distributed
from generate_simulation_data import GenerateSimulationData as g
from my_plots import visualise_simulation, visualise_simulations_comparison, plot_distance_from_goal
from utils import check_dir


def Parse():
    """

    :return args
    """
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')

    parser.add_argument('--gui', action="store_true",
                        help='run simulation using the gui (default: False)')
    parser.add_argument('--myt-quantity', type=int, default=5, metavar='N',
                        help='number of thymios for the simulation (default: 5)')
    parser.add_argument('--simulations', type=int, default=1000, metavar='N',
                        help='number of runs for each simulation (default: 1000)')
    parser.add_argument('--generate-dataset', action="store_true",
                        help='generate the dataset containing the simulations (default: False)')
    parser.add_argument('--plots-dataset', action="store_true",
                        help='generate the plots of regarding the dataset (default: False)')
    parser.add_argument('--check-dataset', action="store_true",
                        help='generate the plots that check the dataset conformity (default: False)')

    parser.add_argument('--controller', default='all', choices=['all', 'learned', 'manual', 'omniscient'],
                        help='choose the controller for the current execution between all, learned, manual and '
                             'omniscient (default: all)')

    parser.add_argument('--train-net', action="store_true",
                        help='train the model  (default: False)')
    parser.add_argument('--model', default='net1', type=str,
                        help='name of the model (default: net1)')
    parser.add_argument('--generate-split', action="store_true",
                        help='generate the indices for the split of the dataset (default: False)')
    parser.add_argument('--net-input', default='prox_values', choices=['prox_values', 'prox_comm'],
                        help='choose the input of the net between prox_values and prox_comm_events (default: '
                             'prox_values)')
    parser.add_argument('--plots-net', action="store_true",
                        help='generate the plots of regarding the model (default: False)')

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

    if args.controller == 'all' or args.controller == 'omniscient':
        if args.generate_dataset:
            print('Generating simulations for %s…' % omniscient_controller)
            g.generate_simulation(runs_dir_omniscient, simulations=args.simulations, controller=omniscient_controller,
                                  myt_quantity=myt_quantity, args=args)

        if args.plots_dataset:
            print('\nGenerating plots for %s…' % omniscient_controller)
            visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 0,
                                 'Distribution simulation %d - %s' % (0, omniscient_controller),
                                 net_input=args.net_input)
            visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 1,
                                 'Distribution simulation %d - %s' % (1, omniscient_controller),
                                 net_input=args.net_input)
            visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 2,
                                 'Distribution simulation %d - %s' % (2, omniscient_controller),
                                 net_input=args.net_input)
            visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 3,
                                 'Distribution simulation %d - %s' % (3, omniscient_controller),
                                 net_input=args.net_input)
            visualise_simulation(runs_dir_omniscient, img_dir_omniscient, 4,
                                 'Distribution simulation %d - %s' % (4, omniscient_controller),
                                 net_input=args.net_input)
            visualise_simulations_comparison(runs_dir_omniscient, img_dir_omniscient,
                                             'Distribution of all simulations - %s' % omniscient_controller,
                                             net_input=args.net_input)

            plot_distance_from_goal(runs_dir_omniscient, img_dir_omniscient, 'Robot distance from goal - %s' %
                                    omniscient_controller, 'distances-from-goal-%s' % omniscient_controller, net_input=args.net_input)

        if args.check_dataset:
            print('Checking conformity of %s dataset…' % omniscient_controller)
            g.check_dataset_conformity(runs_dir_omniscient, img_dir_omniscient, omniscient_controller, net_input=args.net_input)

    # # # # # # #
    manual_controller = "manual-controller"

    dataset_manual = '%dmyts-%s' % (myt_quantity, manual_controller)

    runs_dir_manual = os.path.join('datasets/', dataset_manual)
    img_dir_manual = '%s/images/' % runs_dir_manual

    check_dir(runs_dir_manual)
    check_dir(img_dir_manual)

    if args.controller == 'all' or args.controller == 'manual':
        if args.generate_dataset:
            print('\nGenerating simulations for %s…' % manual_controller)
            g.generate_simulation(runs_dir_manual, simulations=args.simulations, controller=manual_controller,
                                  myt_quantity=myt_quantity, args=args)

        if args.plots_dataset:
            print('\nGenerating plots for %s…' % manual_controller)
            visualise_simulation(runs_dir_manual, img_dir_manual, 0,
                                 'Distribution simulation %d - %s' % (0, manual_controller), net_input=args.net_input)
            visualise_simulation(runs_dir_manual, img_dir_manual, 1,
                                 'Distribution simulation %d - %s' % (1, manual_controller), net_input=args.net_input)
            visualise_simulation(runs_dir_manual, img_dir_manual, 2,
                                 'Distribution simulation %d - %s' % (2, manual_controller), net_input=args.net_input)
            visualise_simulation(runs_dir_manual, img_dir_manual, 3,
                                 'Distribution simulation %d - %s' % (3, manual_controller), net_input=args.net_input)
            visualise_simulation(runs_dir_manual, img_dir_manual, 4,
                                 'Distribution simulation %d - %s' % (4, manual_controller), net_input=args.net_input)
            visualise_simulations_comparison(runs_dir_manual, img_dir_manual,
                                             'Distribution of all simulations - %s' % manual_controller,
                                             net_input=args.net_input)

            plot_distance_from_goal(runs_dir_manual, img_dir_manual, 'Robot distance from goal - %s' %
                                    manual_controller, 'distances-from-goal-%s' % manual_controller,
                                    net_input=args.net_input)

        if args.check_dataset:
            print('Checking conformity of %s dataset…' % manual_controller)
            g.check_dataset_conformity(runs_dir_manual, img_dir_manual, dataset_manual, net_input=args.net_input)

    # # # # # #
    learned_controller = "learned-controller-%s" % args.model

    model_dir = 'models/distributed/%s' % args.model
    model_img = '%s/images/' % model_dir

    check_dir(model_dir)
    check_dir(model_img)

    file = os.path.join('models/distributed/', 'dataset_split.npy')

    if args.controller == 'all' or args.controller == 'learned':
        if args.train_net and args.plots_net:
            print('\nTraining %s and generating plots…' % args.model)
        elif args.train_net and not args.plots_net:
            print('\nTraining %s…' % args.model)
        elif not args.train_net and args.plots_net:
            print('\nGenerating plots for %s…' % args.model)

        run_distributed(file, runs_dir_omniscient, model_dir, model_img, args.model, dataset_omniscient, dataset_manual,
                        train=args.train_net, generate_split=args.generate_split, plots=args.plots_net, net_input=args.net_input)

    # # # # # # #
    dataset_learned = '%dmyts-%s-%s' % (myt_quantity, learned_controller, args.model)

    runs_dir_learned = os.path.join('datasets/', dataset_learned)
    img_dir_learned = '%s/images/' % runs_dir_learned

    check_dir(runs_dir_learned)
    check_dir(img_dir_learned)

    if args.controller == 'all' or args.controller == 'learned':
        if args.generate_dataset:
            print('\nGenerating simulations for %s…' % learned_controller)
            g.generate_simulation(runs_dir_learned, simulations=args.simulations, controller=learned_controller,
                                  myt_quantity=myt_quantity, args=args, model_dir=model_dir, model=args.model)

        if args.plots_dataset:
            print('\nGenerating plots for %s…' % learned_controller)
            visualise_simulation(runs_dir_learned, img_dir_learned, 0,
                                 'Distribution simulation %d - %s' % (0, learned_controller), net_input=args.net_input)
            visualise_simulation(runs_dir_learned, img_dir_learned, 1,
                                 'Distribution simulation %d - %s' % (1, learned_controller), net_input=args.net_input)
            visualise_simulation(runs_dir_learned, img_dir_learned, 2,
                                 'Distribution simulation %d - %s' % (2, learned_controller), net_input=args.net_input)
            visualise_simulation(runs_dir_learned, img_dir_learned, 3,
                                 'Distribution simulation %d - %s' % (3, learned_controller), net_input=args.net_input)
            visualise_simulation(runs_dir_learned, img_dir_learned, 4,
                                 'Distribution simulation %d - %s' % (4, learned_controller), net_input=args.net_input)
            visualise_simulations_comparison(runs_dir_learned, img_dir_learned,
                                             'Distribution of all simulations - %s' % learned_controller,
                                             net_input=args.net_input)

            plot_distance_from_goal(runs_dir_learned, img_dir_learned, 'Robot distance from goal - %s' %
                                    learned_controller, 'distances-from-goal-%s' % learned_controller,
                                    net_input=args.net_input)

        if args.check_dataset:
            print('Checking conformity of %s dataset…' % learned_controller)
            g.check_dataset_conformity(runs_dir_learned, img_dir_learned, dataset_learned, net_input=args.net_input)

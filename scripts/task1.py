import argparse
import os

from utils import directory_for_dataset, directory_for_model


def parse_args():
    """

    :return args
    """
    parser = argparse.ArgumentParser(description='Simulation of robot swarms for learning communication-aware coordination - Task 1')

    parser.add_argument('--gui', action="store_true",
                        help='run simulation using the gui (default: False)')
    parser.add_argument('--myt-quantity', type=int, default=5, metavar='N',
                        help='number of thymios for the simulation (default: 5)')
    parser.add_argument('--avg-gap', type=int, default=8, metavar='N',
                        help='average gap distance between thymios (default: 8)')

    parser.add_argument('--n-simulations', type=int, default=1000, metavar='N',
                        help='number of runs for each simulation (default: 1000)')
    parser.add_argument('--generate-dataset', action="store_true",
                        help='generate the dataset containing the simulations (default: False)')
    parser.add_argument('--generate-split', action="store_true",
                        help='generate the indices for the split of the dataset (default: False)')

    parser.add_argument('--plots-dataset', action="store_true",
                        help='generate the plots of regarding the dataset (default: False)')
    parser.add_argument('--check-dataset', action="store_true",
                        help='generate the plots that check the dataset conformity (default: False)')

    parser.add_argument('--controller', default='all', choices=['all', 'learned', 'manual', 'omniscient'],
                        help='choose the controller for the current execution between all, learned, manual and '
                             'omniscient (default: all)')

    parser.add_argument('--dataset-folder', default='datasets', type=str,
                        help='name of the directory containing the datasets (default: datasets)')
    parser.add_argument('--dataset', default='all', type=str,
                        help='choose the datasets to use in the current execution (default: all)')

    parser.add_argument('--models-folder', default='models', type=str,
                        help='name of the directory containing the models (default: models)')
    parser.add_argument('--model-type', default='distributed', type=str,
                        help='name of the sub-directory containing the models (default: distributed)')
    parser.add_argument('--model', default='net1', type=str,
                        help='name of the model (default: net1)')

    parser.add_argument('--train-net', action="store_true",
                        help='train the model  (default: False)')

    parser.add_argument('--net-input', default='prox_values', choices=['prox_values', 'prox_comm', 'all_sensors'],
                        help='choose the input of the net between prox_values and prox_comm_events (default: '
                             'prox_values)')
    parser.add_argument('--plots-net', action="store_true",
                        help='generate the plots of regarding the model (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    if args.dataset == 'all':
        datasets = [f.path for f in os.scandir(args.dataset_folder) if f.is_dir()]
    else:
        dataset = os.path.join(args.dataset_folder, args.net_input, args.dataset)
        datasets = [dataset]

    if args.controller == 'all':
        controllers = ['omniscient', 'learned', 'manual']
    else:
        controllers = [args.controller]

    runs_dir_omniscient = os.path.join(datasets[0], 'omniscient')
    runs_dir_manual = os.path.join(datasets[0], 'manual')
    runs_dir_learned = os.path.join(datasets[0], 'learned')

    for d in datasets:
        for c in controllers:
            run_dir, run_img_dir, run_video_dir = directory_for_dataset(d, c)
            model_dir, model_img_dir, model_video_dir, metrics_path = directory_for_model(args)

            myt_quantity = args.myt_quantity

            if args.generate_dataset:
                from generate_simulation_data import GenerateSimulationData as sim

                print('Generating %s simulations for %s %s controller…' % (args.n_simulations, d, c))

                if c == 'learned':
                    sim.generate_simulation(run_dir=run_dir, n_simulations=args.n_simulations, controller=c,
                                            myt_quantity=myt_quantity, args=args.gui, model_dir=model_dir, model=args.model)
                else:
                    sim.generate_simulation(run_dir=run_dir, n_simulations=args.n_simulations, controller=c,
                                            myt_quantity=myt_quantity, args=args.gui)

            if args.plots_dataset:
                from my_plots import visualise_simulation, visualise_simulations_comparison, plot_distance_from_goal


                print('Generating plots for %s %s controller…' % (d, c))

                if not args.net_input == 'all_sensors':
                    for i in range(5):
                        visualise_simulation(run_dir, run_img_dir, i, 'Distribution simulation %d - %s' % (i, c),
                                             net_input=args.net_input)
                    visualise_simulations_comparison(run_dir, run_img_dir, 'Distribution of all simulations - %s' % c,
                                                     net_input=args.net_input)

                plot_distance_from_goal(run_dir, run_img_dir, 'Robot distance from goal - %s' % c,
                                        'distances-from-goal-%s' % c, net_input=args.net_input)

            if args.check_dataset:
                print('\nChecking conformity of %s dataset…' % c)
                sim.check_dataset_conformity(run_dir, run_img_dir, c, net_input=args.net_input)

            if args.train_net:
                from distributed import run_distributed
                file = os.path.join(run_dir, 'dataset_split.npy')

                print('\nTraining %s…' % args.model)
                # FIXME
                run_distributed(file, runs_dir_omniscient, model_dir, model_img_dir, args.model, 'omniscient', 'manual',
                                train=args.train_net, generate_split=args.generate_split, plots=args.plots_net,
                                net_input=args.net_input, avg_gap=args.avg_gap)

                if args.plot_net:
                    from my_plots import plot_sensing_timestep
                    print('\nGenerating plots for %s…' % args.model)
                    if not args.net_input == 'all_sensors':
                        plot_sensing_timestep(runs_dir_omniscient, model_img_dir, net_input=args.net_input, model=args.model)

    from my_plots import plot_compared_distance_from_goal
    # Comparison among all datasets
    runs_img_dir = os.path.join(datasets[0], 'images')
    plot_compared_distance_from_goal(runs_dir_omniscient, runs_dir_manual, runs_dir_learned, runs_img_dir,
                                     'Robot distances from goal', 'distances-from-goal', net_input=args.net_input)

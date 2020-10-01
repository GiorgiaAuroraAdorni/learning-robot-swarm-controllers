import argparse
import os

from utils.utils import directory_for_dataset, directory_for_model


def parse_args():
    """
    Imitation Learning - Extension to an arbitrary number of agents

    usage: task1_extensions.py [--args]

    :return args:
                  --help                Show this help message and exit
                  --gui                 Run simulation using the gui (default: False)
                  --n-simulations N     Number of runs for each simulation (default: 1000)
                  --task TASK
                                        Choose the task to perform in the current execution
                                        between task1 and task2 (default: task1)
                  --myt-quantity MYT_QUANTITY
                                        Number of thymios for the simulation (default: variable)
                  --avg-gap AVG_GAP
                                        Average gap distance between thymios (default: variable)
                  --generate-dataset    Generate the dataset containing the simulations
                                        (default: False)
                  --generate-split      Generate the indices for the split of the dataset
                                        (default: False)
                  --plots-dataset       Generate the plots of regarding the dataset
                                        (default: False)
                  --check-dataset       Generate the plots that check the dataset conformity
                                        (default: False)
                  --compare-all         Generate plots that compare all the experiments in
                                        terms of distance from goal (default: False)
                  --generate-animations
                                        Generate animations that compare the controllers
                                        (default: False)
                  --controller CONTROLLER
                                        Choose the controller for the current execution.
                                        Usually between all, learned, manual and omniscient
                                        (default: all)
                  --dataset-folder DATASET_FOLDER
                                        Name of the directory containing the datasets
                                        (default: datasets)
                  --dataset DATASET     Choose the datasets to use in the current execution
                                        (default: all)
                  --models-folder MODELS_FOLDER
                                        Name of the directory containing the models
                                        (default: models)
                  --model-type MODEL_TYPE
                                        Name of the sub-directory containing the models
                                        (default: distributed)
                  --model MODEL         Name of the model (default: net1)
                  --train-net           Train the model (default: False)
                  --save-net            Save the model in onnx format (default: False)
                  --net-input SENSING
                                        Choose the input of the net between prox_values,
                                        prox_comm or all_sensors (default: all_sensors)
                  --plots-net           Generate the plots of regarding the model
                                        (default: False)

    """
    parser = argparse.ArgumentParser(description='Imitation Learning - Extension to an arbitrary number of agents')

    parser.add_argument('--gui', action="store_true",
                        help='Run simulation using the gui (default: False)')
    parser.add_argument('--n-simulations', type=int, default=1000, metavar='N',
                        help='Number of runs for each simulation (default: 1000)')
    parser.add_argument('--task', default='task1', choices=['task1', 'task2'],
                        help='Choose the task to perform in the current execution between task1 and task2 (default: task1)')

    parser.add_argument('--myt-quantity', default='variable', type=str,
                        help='Number of thymios for the simulation (default: variable)')
    parser.add_argument('--avg-gap', default='variable', type=str,
                        help='Average gap distance between thymios (default: variable)')

    parser.add_argument('--generate-dataset', action="store_true",
                        help='Generate the dataset containing the simulations (default: False)')
    parser.add_argument('--generate-split', action="store_true",
                        help='Generate the indices for the split of the dataset (default: False)')

    parser.add_argument('--plots-dataset', action="store_true",
                        help='Generate the plots of regarding the dataset (default: False)')
    parser.add_argument('--check-dataset', action="store_true",
                        help='Generate the plots that check the dataset conformity (default: False)')
    parser.add_argument('--compare-all', action="store_true",
                        help='Generate plots that compare all the experiments in terms of distance from goal ('
                             'default: False)')
    parser.add_argument('--generate-animations', action="store_true",
                        help='Generate animations that compare the controllers (default: False)')

    parser.add_argument('--controller', default='all', type=str,
                        help='Choose the controller for the current execution. Usually between all, learned, '
                             'manual and omniscient (default: all)')

    parser.add_argument('--dataset-folder', default='datasets', type=str,
                        help='Name of the directory containing the datasets (default: datasets)')
    parser.add_argument('--dataset', default='all', type=str,
                        help='Choose the datasets to use in the current execution (default: all)')

    parser.add_argument('--models-folder', default='models', type=str,
                        help='Name of the directory containing the models (default: models)')
    parser.add_argument('--model-type', default='distributed', type=str,
                        help='Name of the sub-directory containing the models (default: distributed)')
    parser.add_argument('--model', default='net1', type=str, help='Name of the model (default: net1)')

    parser.add_argument('--train-net', action='store_true', help='Train the model (default: False)')
    parser.add_argument('--save-net', action='store_true', help='Save the model in onnx format (default: False)')

    parser.add_argument('--net-input', default='all_sensors', choices=['prox_values', 'prox_comm', 'all_sensors'],
                        help='Choose the input of the net between prox_values and prox_comm_events (default: '
                             'all_sensors)')
    parser.add_argument('--plots-net', action="store_true",
                        help='Generate the plots of regarding the model (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    d = os.path.join(args.dataset_folder, args.task, args.net_input, args.dataset)

    if args.controller == 'all':
        controllers = ['omniscient', 'learned_distributed', 'learned_communication', 'manual']
    else:
        controllers = [args.controller]

    runs_dir_omniscient = os.path.join(d, 'omniscient')
    runs_dir_manual = os.path.join(d, 'manual')
    runs_dir_learned = os.path.join(d, 'learned')
    runs_dir_learned_dist = os.path.join(d, 'learned_distributed')
    runs_dir_learned_comm = os.path.join(d, 'learned_communication')

    for c in controllers:
        run_dir, run_img_dir, run_video_dir = directory_for_dataset(d, c)
        model_dir, model_img_dir, model_video_dir, metrics_path = directory_for_model(args)

        if args.model_type == 'communication':
            communication = True
        elif args.model_type == 'distributed':
            communication = False
        else:
            raise ValueError('Invalid value for model_type')

        if args.generate_dataset:
            from generate_simulation_data import GenerateSimulationData as sim

            print('Generating %s simulations for %s %s controller…' % (args.n_simulations, d, c))

            if c == 'learned':
                sim.generate_simulation(run_dir=run_dir, n_simulations=args.n_simulations, controller=c,
                                        args=args, model_dir=model_dir, model=args.model, communication=communication)
            else:
                sim.generate_simulation(run_dir=run_dir, n_simulations=args.n_simulations, controller=c,
                                        args=args, communication=communication)

        if args.plots_dataset:
            from utils.my_plots import visualise_simulation_over_time_all_sensors, plot_compared_distance_compressed, \
                                       thymio_quantity_distribution, visualise_position_over_time, \
                                       visualise_control_over_time, plot_predicted_message

            print('Generating plots for %s %s controller…' % (d, c))
            runs = [0, 1, 2, 3, 4, 5, 251, 634]
            # for i in runs:
            #     visualise_simulation_over_time_all_sensors(run_dir, run_img_dir, i,
            #                                                'Thymio positions over time - Simulation run %d - %s %s'
            #                                                % (i, args.net_input, c))
            #
            # # FIXME substitute controllers with datasets
            # datasets = ['omniscient']
            # plot_compared_distance_compressed([run_dir], run_img_dir, controllers,
            #                                   'Robot distances from goal - %s' % (args.net_input),
            #                                   'distances-from-goal-compressed')
            #
            # plot_compared_distance_compressed([run_dir], run_img_dir, controllers,
            #                                   'Robot distances from goal - %s' % (args.net_input),
            #                                   'distances-from-goal-absolute-compressed', absolute=False)
            if args.myt_quantity != 'variable':
                visualise_position_over_time(run_dir, run_img_dir, 'position-overtime-%s' % c)
            # visualise_control_over_time(run_dir, run_img_dir, 'control-overtime-%s' % c)

            thymio_quantity_distribution(run_dir, run_img_dir, 'Thymio quantity distribution - %s' % (args.net_input),
                                         'thymio-quantity-distribution')

            if communication:
                from utils.my_plots import visualise_communication_simulation, visualise_communication_vs_control, \
                                           visualise_communication_vs_distance
                for i in range(5):
                    visualise_communication_simulation(run_dir, run_img_dir, i,
                                                       'Simulation run %d - %s %s - communication' %
                                                       (i, args.net_input, c))

                visualise_communication_vs_control(run_dir, run_img_dir,
                                                   'Communication vs Control - %s %s' % (args.net_input, c))

                visualise_communication_vs_distance(run_dir, run_img_dir,
                                                    'Communication vs Distance from goal - %s %s' % (args.net_input, c))
                    plot_predicted_message(run_dir, run_img_dir, i)


        if args.check_dataset:
            from generate_simulation_data import GenerateSimulationData as sim
            print('\nChecking conformity of %s %s dataset…' % (d, c))
            sim.check_dataset_conformity(run_dir, run_img_dir, 'Dataset - %s %s' % (args.net_input, c), c,
                                         net_input=args.net_input, communication=communication)

        if args.train_net or args.plots_net:
            from utils.utils import prepare_dataset

            indices = prepare_dataset(run_dir, args.generate_split, args.n_simulations)
            file_losses = os.path.join(model_dir, 'losses.npy')

            if args.train_net:
                from network_training_distribute import network_train
                network_train(indices, file_losses, runs_dir_omniscient, model_dir, args.model, communication,
                              net_input=args.net_input, save_net=args.save_net)

            if args.plots_net:
                from network_evaluation import network_evaluation

                network_evaluation(indices, file_losses, runs_dir_omniscient, model_dir, args.model, model_img_dir,
                                   'omniscient', 'manual', communication, net_input=args.net_input, task=args.task)

    if args.compare_all:
        from utils.my_plots import plot_compared_distance_compressed, test_controller_given_init_positions

        print('\nGenerating comparison plots among all datasets of type %s…' % (args.net_input))
        runs_img_dir = os.path.join(d, 'images')
        
        dataset_folders_comm = [runs_dir_omniscient, runs_dir_manual, runs_dir_learned_dist, runs_dir_learned_comm]
        datasets_comm = ['omniscient', 'manual', 'distributed', 'communication']

        dataset_folders_dist = [runs_dir_omniscient, runs_dir_manual, runs_dir_learned_dist]
        datasets_dist = ['omniscient', 'manual', 'distributed']
        
        plot_compared_distance_compressed(dataset_folders_dist, runs_img_dir, datasets_dist,
                                         'Robot distances from goal', 'distances-from-goal-compressed-distributed')
        
        plot_compared_distance_compressed(dataset_folders_comm, runs_img_dir, datasets_comm,
                                         'Robot distances from goal', 'distances-from-goal-compressed-communication')

        # Evaluate the learned controllers by passing a specific initial position configuration and compare them with
        # the omniscient and the manual controllers
        test_controller_given_init_positions(runs_img_dir, args.model, args.net_input)

    if args.generate_animations:
        from utils.my_plots import animate_simulation, plot_simulations, visualise_position_over_time
        from utils.utils import generate_fake_simulations, generate_init_positions, check_dir
        import numpy as np
        animations_dir = os.path.join(d, 'animations')
        check_dir(animations_dir)

        plots_dir = os.path.join(d, 'plots')
        check_dir(plots_dir)

        # Create a simulation for each of the controller using the same initial position
        if args.myt_quantity == 'variable':
            myt_quantities = np.arange(5, 11)
        else:
            myt_quantities = [int(args.myt_quantity)]

        for N in myt_quantities:
            dir = os.path.join(plots_dir, 'N%d' % N)
            check_dir(dir)
            _ = generate_fake_simulations(dir, args.model, N, simulations=100)
            out_dirs = [os.path.join(dir, 'omniscient'), os.path.join(dir, 'manual'), os.path.join(dir, 'distributed'), os.path.join(dir, 'communication')]

            for c in out_dirs:
                controller = os.path.basename(os.path.normpath(c))
                img_dir = os.path.join(c, 'images')
                check_dir(img_dir)
                visualise_position_over_time(c, img_dir, 'position-overtime-%s' % controller)

            # FIXME
            # dir = os.path.join(animations_dir, 'N%d' % N)
            # check_dir(dir)
            # out_dirs = generate_fake_simulations(dir, args.model, N, simulations=1)
            #
            # for c in out_dirs:
            #     controller = os.path.basename(os.path.normpath(c))
            #     img_dir = os.path.join(c, 'images')
            #     check_dir(img_dir)
            #
            #     animate_simulation(out_dirs, N)
            #     plot_simulations(out_dirs, N)

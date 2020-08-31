import argparse
import os

from utils.utils import directory_for_dataset, directory_for_model


def parse_args():
    """

    Imitation Learning - Task 2 with Communication

    usage: task2.py [--args]

    :return args:
                  --help                Show this help message and exit
                  --gui                 Run simulation using the gui (default: False)
                  --n-simulations N     Number of runs for each simulation (default: 1000)
                  --task TASK
                                        Choose the task to perform in the current execution
                                        between task1 and task2 (default: task2)
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
    parser = argparse.ArgumentParser(description='Imitation Learning - Task 2 with Communication')

    parser.add_argument('--gui', action="store_true",
                        help='Run simulation using the gui (default: False)')
    parser.add_argument('--n-simulations', type=int, default=1000, metavar='N',
                        help='Number of runs for each simulation (default: 1000)')
    parser.add_argument('--task', default='task2', choices=['task1', 'task2'],
                        help='Choose the task to perform in the current execution between task1 and task2 (default: task2)')

    parser.add_argument('--myt-quantity', type=str, default='variable',
                        help='Number of thymios for the simulation (default: variable)')
    parser.add_argument('--avg-gap', type=str, default='variable',
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
                        help='Generate plots that compare all the experiments in terms of distance from goal '
                             '(default: False)')
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
    parser.add_argument('--model-type', default='communication', type=str,
                        help='Name of the sub-directory containing the models (default: communication)')
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

    d = os.path.join(args.dataset_folder, args.task, args.dataset)

    if args.controller == 'all':
        controllers = ['omniscient', 'learned', 'manual']
    else:
        controllers = [args.controller]

    runs_dir_omniscient = os.path.join(d, 'omniscient')
    runs_dir_manual = os.path.join(d, 'manual')
    runs_dir_learned = os.path.join(d, 'learned')

    for c in controllers:
        run_dir, run_img_dir, run_video_dir = directory_for_dataset(d, c)
        model_dir, model_img_dir, model_video_dir, metrics_path = directory_for_model(args)

        myt_quantity = args.myt_quantity

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
                                        myt_quantity=None, args=args, model_dir=model_dir, model=args.model,
                                        communication=communication)
            else:
                sim.generate_simulation(run_dir=run_dir, n_simulations=args.n_simulations, controller=c,
                                        myt_quantity=None, args=args, communication=communication)

        if args.plots_dataset:
            from utils.my_plots import thymio_quantity_distribution, visualise_communication_simulation, \
                                       visualise_communication_vs_control, visualise_communication_vs_distance, \
                                       plot_predicted_colour, plot_predicted_message

            print('Generating plots for %s %s controller…' % (d, c))
            thymio_quantity_distribution(run_dir, run_img_dir, 'Thymio quantity distribution - %s' % (args.net_input),
                                         'thymio-quantity-distribution')

            if c == 'omniscient':
                target = 'goal_colour'
            else:
                target = 'colour'

            for i in range(5):
                plot_predicted_colour(run_dir, run_img_dir, i, target=target)

            if not c == 'omniscient':
                for i in range(5):
                    plot_predicted_message(run_dir, run_img_dir, i)

                    visualise_communication_simulation(run_dir, run_img_dir, i,
                                                       'Simulation run %d - %s %s - communication' %
                                                       (i, args.net_input, c))

                visualise_communication_vs_control(run_dir, run_img_dir,
                                                   'Communication vs Control - %s %s' % (args.net_input, c))

                visualise_communication_vs_distance(run_dir, run_img_dir,
                                                    'Communication vs Distance from goal - %s %s' % (args.net_input, c))

        if args.train_net or args.plots_net:
            from utils.utils import prepare_dataset

            indices = prepare_dataset(run_dir, args.generate_split, args.n_simulations)
            file_losses = os.path.join(model_dir, 'losses.npy')

            if args.train_net:
                from network_training_colour import network_train
                network_train(indices, file_losses, runs_dir_omniscient, model_dir, args.model, communication,
                              net_input=args.net_input, save_net=args.save_net)

            if args.plots_net:
                from network_evaluation import network_evaluation
                network_evaluation(indices, file_losses, runs_dir_omniscient, model_dir, args.model, model_img_dir,
                                   'omniscient', 'manual', communication, net_input=args.net_input, task=args.task,
                                   runs_dir_manual=runs_dir_manual, runs_dir_learned=runs_dir_learned)

    if args.compare_all:
        from utils.my_plots import plot_compared_colour_error

        print('\nGenerating comparison plots among all datasets of type %s…' % (args.net_input))
        runs_img_dir = os.path.join(d, 'images')
        dataset_folders = [runs_dir_omniscient, runs_dir_manual, runs_dir_learned]
        datasets = ['omniscient', 'manual', 'learned']

        plot_compared_colour_error(dataset_folders, runs_img_dir, datasets, 'colours-errors-compressed')
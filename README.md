# Robot swarms simulation for learning communication aware coordination
> Master thesis project that simulate robot swarms for learning communication-aware coordination. 
>
> @ USI 19/20.
>
> See <https://drive.google.com/drive/folders/1KCUJn06g2zrhAr3wk5BOAC7EoNM_4XOr?usp=sharing> for datasets, models and report.

#### Contributors

**Giorgia Adorni** - giorgia.adorni@usi.ch  [GiorgiaAuroraAdorni](https://github.com/GiorgiaAuroraAdorni)

#### Prerequisites

- Python 3
- Enki
- PyTorch

#### Installation

To install Enki follow the following instructions: https://jeguzzi.github.io/enki/intro.html.

Clone our repository and install the requirements

```sh
$ git clone https://github.com/GiorgiaAuroraAdorni/learning-robot-swarm-controllers
$ cd learning-robot-swarm-controllers
$ pip install -r requirements.txt
```

#### Usage

To receive help on how to run the scripts (for example for task1), execute:

```sh
$ python source/task1.py --help

> usage: task1.py [-h] [--gui] [--myt-quantity N] [--n-simulations N]
                  [--task {task1,task2}] [--avg-gap N] [--generate-dataset]
                  [--generate-split] [--plots-dataset] [--check-dataset]
                  [--compare-all] [--controller CONTROLLER]
                  [--dataset-folder DATASET_FOLDER] [--dataset DATASET]
                  [--models-folder MODELS_FOLDER] [--model-type MODEL_TYPE]
                  [--model MODEL] [--train-net] [--save-net]
                  [--net-input {prox_values,prox_comm,all_sensors}]
                  [--plots-net]

Imitation Learning - Distributed Controller + Communication

optional arguments:
  -h, --help            show this help message and exit
  --gui                 Run simulation using the gui (default: False)
  --myt-quantity N      Number of thymios for the simulation (default: 5)
  --n-simulations N     Number of runs for each simulation (default: 1000)
  --task {task1,task2}  Choose the task to perform in the current execution
                        between task1 and task2 (default: task1)
  --avg-gap N           Average gap distance between thymios (default: 8)
  --generate-dataset    Generate the dataset containing the simulations
                        (default: False)
  --generate-split      Generate the indices for the split of the dataset
                        (default: False)
  --plots-dataset       Generate the plots of regarding the dataset (default:
                        False)
  --check-dataset       Generate the plots that check the dataset conformity
                        (default: False)
  --compare-all         Generate plots that compare all the experiments in
                        terms of distance from goal (default: False)
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
                        Name of the directory containing the models (default:
                        models)
  --model-type MODEL_TYPE
                        Name of the sub-directory containing the models
                        (default: distributed)
  --model MODEL         Name of the model (default: net1)
  --train-net           Train the model (default: False)
  --save-net            Save the model in onnx format (default: False)
  --net-input {prox_values,prox_comm,all_sensors}
                        Choose the input of the net between prox_values,
                        prox_comm or all_sensors (default: prox_values)
  --plots-net           Generate the plots of regarding the model (default:
                        False)

```


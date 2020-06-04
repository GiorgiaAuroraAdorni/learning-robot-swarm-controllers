# Robot swarms simulation for learning communication aware coordination
> Master thesis project that simulate robot swarms for learning communication-aware coordination. 
>
> @ USI 19/20.
>
> See <https://drive.google.com/drive/folders/1mUF_uHNvKKEePV2O_CP-Pd_aTXNxHpZR> for datasets, models and report.

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

To receive help on how to run the script, execute:

```sh
$ python scripts/task1.py --help

> usage: task1.py [-h] [--gui] [--myt-quantity N] [--simulations N]
                  [--generate-dataset] [--plots-dataset] [--check-dataset]
                  [--avg-gap N] [--controller {all,learned,manual,omniscient}]
                  [--dataset-folder DATASET_FOLDER] [--train-net]
                  [--model MODEL] [--generate-split]
                  [--net-input {prox_values,prox_comm,all_sensors}]
                  [--plots-net]

Simulation of robot swarms for learning communication-aware coordination -
Task 1

optional arguments:
  -h, --help            show this help message and exit
  --gui                 run simulation using the gui (default: False)
  --myt-quantity N      number of thymios for the simulation (default: 5)
  --simulations N       number of runs for each simulation (default: 1000)
  --generate-dataset    generate the dataset containing the simulations
                        (default: False)
  --plots-dataset       generate the plots of regarding the dataset (default:
                        False)
  --check-dataset       generate the plots that check the dataset conformity
                        (default: False)
  --avg-gap N           average gap distance between thymios (default: 8)
  --controller {all,learned,manual,omniscient}
                        choose the controller for the current execution
                        between all, learned, manual and omniscient (default:
                        all)
  --dataset-folder DATASET_FOLDER
                        name of the directory containing the datasets
                        (default: datasets/)
  --train-net           train the model (default: False)
  --model MODEL         name of the model (default: net1)
  --generate-split      generate the indices for the split of the dataset
                        (default: False)
  --net-input {prox_values,prox_comm,all_sensors}
                        choose the input of the net between prox_values and
                        prox_comm_events (default: prox_values)
  --plots-net           generate the plots of regarding the model (default:
                        False)
```

##### 
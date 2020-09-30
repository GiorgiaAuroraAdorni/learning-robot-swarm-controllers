#!/bin/bash

# Compare all
python3 source/task1.py --model-type communication --compare-all --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model 1
python3 source/task1.py --model-type communication --compare-all --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model 2
python3 source/task1.py --model-type communication --compare-all --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model 3
python3 source/task1.py --model-type communication --compare-all --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model 4
python3 source/task1.py --model-type communication --compare-all --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model 5
python3 source/task1.py --model-type communication --compare-all --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model 6
python3 source/task1.py --model-type communication --compare-all --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model 7
python3 source/task1.py --model-type communication --compare-all --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model 8
python3 source/task1.py --model-type communication --compare-all --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model 9

# Only plots for the datasets
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net-c1
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net-c2
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net-c3
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net-c4
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net-c5
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net-c6
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net-c7
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net-c8
python3 source/task1.py --model-type communication --controller learned_communication --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net-c9


# Experiment A
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net-c1
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net-c1


# Experiment A2
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net-c2
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net-c2


# Experiment A3
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net-c3
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net-c3


# Experiment B
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net-c4
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net-c4


# Experiment C
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net-c5
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net-c5


# Experiment D
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net-c6
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net-c6


# Experiment E
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net-c7
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net-c7


# Experiment F
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net-c8
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net-c8


# Experiment G
python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --save-net --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net-c9
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net-c9

#!/bin/bash


# Only plots for the datasets
python3 source/task1.py --controller all --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net-d1
python3 source/task1.py --controller all --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net-d2
python3 source/task1.py --controller all --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net-d3
python3 source/task1.py --controller all --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net-d4
python3 source/task1.py --controller all --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net-d5
python3 source/task1.py --controller all --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net-d6
python3 source/task1.py --controller all --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net-d7
python3 source/task1.py --controller all --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net-d8
python3 source/task1.py --controller all --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net-d9


# Only plots for the models
python3 source/task1.py --controller omniscient --plots-net --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net-d1
python3 source/task1.py --controller omniscient --plots-net --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net-d2
python3 source/task1.py --controller omniscient --plots-net --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net-d3
python3 source/task1.py --controller omniscient --plots-net --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net-d4
python3 source/task1.py --controller omniscient --plots-net --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net-d5
python3 source/task1.py --controller omniscient --plots-net --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net-d6
python3 source/task1.py --controller omniscient --plots-net --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net-d7
python3 source/task1.py --controller omniscient --plots-net --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net-d8
python3 source/task1.py --controller omniscient --plots-net --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net-d9


# Experiment A
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller omniscient --plots-net --train-net --save-net --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net-d1
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net-d1


# Experiment A2
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net-d2
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net-d2


# Experiment A3
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net-d3
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net-d3


# Experiment B
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net-d4
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net-d4


# Experiment C
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net-d5
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net-d5


# Experiment D
python3 source/task1.py --controller omniscient --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net-d6
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net-d6


# Experiment E
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net-d7
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net-d7


# Experiment F
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net-d8
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net-d8


# Experiment G
python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net-d9
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net-d9

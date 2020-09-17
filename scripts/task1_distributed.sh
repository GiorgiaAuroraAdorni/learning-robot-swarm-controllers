#!/bin/bash


# Experiment A

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net1
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net1
# Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8
# Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net1


# Experiment A2

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net2
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net2
## Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input prox_values --avg-gap 13 --dataset avg_gap-13
## Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input prox_values --avg-gap 13 --dataset avg_gap-13 --model net2


# Experiment A3

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net3
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net3
## Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input prox_values --avg-gap 24 --dataset avg_gap-24
## Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input prox_values --avg-gap 24 --dataset avg_gap-24 --model net3


# Experiment B

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net4
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net4
## Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8
## Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net4


# Experiment C

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net5
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net5
## Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13
## Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net5


# Experiment D

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net6
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net6
## Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input prox_comm --avg-gap 24 --dataset avg_gap-24
## Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input prox_comm --avg-gap 24 --dataset avg_gap-24 --model net6


# Experiment E

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net7
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net7
## Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8
## Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net7


# Experiment F

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net8
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net8
## Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13
## Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net8


# Experiment G

python3 source/task1.py --controller omniscient --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller omniscient --generate-split --plots-net --train-net --save-net --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net9
python3 source/task1.py --controller manual --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24
python3 source/task1.py --controller learned --generate-dataset --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net9
## Only plots for the datasets
#python3 source/task1.py --controller all --plots-dataset --net-input all_sensors --avg-gap 24 --dataset avg_gap-24
## Only plots for the model
#python3 source/task1.py --controller omniscient --plots-net --net-input all_sensors --avg-gap 24 --dataset avg_gap-24 --model net9

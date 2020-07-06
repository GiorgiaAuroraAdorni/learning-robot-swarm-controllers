#!/bin/bash


# Experiment A

python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net1
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net1
python3 source/task1.py --model-type communication --compare-all --net-input prox_values --avg-gap 8 --dataset avg_gap-8


# Experiment B

python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net2
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net2
python3 source/task1.py --model-type communication --compare-all --net-input prox_comm --avg-gap 8 --dataset avg_gap-8

# Experiment C

python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net3
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net3
python3 source/task1.py --model-type communication --compare-all --net-input prox_comm --avg-gap 13 --dataset avg_gap-13


# Experiment D

python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --net-input prox_comm --avg-gap 25 --dataset avg_gap-25 --model net4
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 25 --dataset avg_gap-25 --model net4
python3 source/task1.py --model-type communication --compare-all --net-input prox_comm --avg-gap 25 --dataset avg_gap-25


# Experiment E

python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net5
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net5
python3 source/task1.py --model-type communication --compare-all --net-input all_sensors --avg-gap 8 --dataset avg_gap-8


# Experiment F

python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net6
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net6
python3 source/task1.py --model-type communication --compare-all --net-input all_sensors --avg-gap 13 --dataset avg_gap-13


# Experiment G

python3 source/task1.py --model-type communication --controller omniscient --plots-net --train-net --net-input all_sensors --avg-gap 25 --dataset avg_gap-25 --model net7
python3 source/task1.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 25 --dataset avg_gap-25 --model net7
python3 source/task1.py --model-type communication --compare-all --net-input all_sensors --avg-gap 25 --dataset avg_gap-25
#!/bin/bash


# Experiment A

python3 source/run_distributed_no_communication.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8
python3 source/run_distributed_no_communication.py --controller omniscient --generate-split --plots-net --train-net --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net1
python3 source/run_distributed_no_communication.py --controller manual --generate-dataset --plots-dataset --check-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8
python3 source/run_distributed_no_communication.py --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_values --avg-gap 8 --dataset avg_gap-8 --model net1
python3 source/run_distributed_no_communication.py --compare-all --net-input prox_values --avg-gap 8 --dataset avg_gap-8

# Experiment B

python3 source/run_distributed_no_communication.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8
python3 source/run_distributed_no_communication.py --controller omniscient --generate-split --plots-net --train-net --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net2
python3 source/run_distributed_no_communication.py --controller manual --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8
python3 source/run_distributed_no_communication.py --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 8 --dataset avg_gap-8 --model net2
python3 source/run_distributed_no_communication.py --compare-all --net-input prox_comm --avg-gap 8 --dataset avg_gap-8


# Experiment C

python3 source/run_distributed_no_communication.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13
python3 source/run_distributed_no_communication.py --controller omniscient --generate-split --plots-net --train-net --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net3
python3 source/run_distributed_no_communication.py --controller manual --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13
python3 source/run_distributed_no_communication.py --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 13 --dataset avg_gap-13 --model net3
python3 source/run_distributed_no_communication.py --compare-all --net-input prox_comm --avg-gap 13 --dataset avg_gap-13


# Experiment D

python3 source/run_distributed_no_communication.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 25 --dataset avg_gap-25
python3 source/run_distributed_no_communication.py --controller omniscient --generate-split --plots-net --train-net --net-input prox_comm --avg-gap 25 --dataset avg_gap-25 --model net4
python3 source/run_distributed_no_communication.py --controller manual --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 25 --dataset avg_gap-25
python3 source/run_distributed_no_communication.py --controller learned --generate-dataset --plots-dataset --check-dataset --net-input prox_comm --avg-gap 25 --dataset avg_gap-25 --model net4
python3 source/run_distributed_no_communication.py --compare-all --net-input prox_comm --avg-gap 25 --dataset avg_gap-25


# Experiment E

python3 source/run_distributed_no_communication.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8
python3 source/run_distributed_no_communication.py --controller omniscient --generate-split --plots-net --train-net --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net5
python3 source/run_distributed_no_communication.py --controller manual --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8
python3 source/run_distributed_no_communication.py --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 8 --dataset avg_gap-8 --model net5
python3 source/run_distributed_no_communication.py --compare-all --net-input all_sensors --avg-gap 8 --dataset avg_gap-8


# Experiment F

python3 source/run_distributed_no_communication.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13
python3 source/run_distributed_no_communication.py --controller omniscient --generate-split --plots-net --train-net --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net6
python3 source/run_distributed_no_communication.py --controller manual --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13
python3 source/run_distributed_no_communication.py --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 13 --dataset avg_gap-13 --model net6
python3 source/run_distributed_no_communication.py --compare-all --net-input all_sensors --avg-gap 13 --dataset avg_gap-13


# Experiment G

python3 source/run_distributed_no_communication.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 25 --dataset avg_gap-25
python3 source/run_distributed_no_communication.py --controller omniscient --generate-split --plots-net --train-net --net-input all_sensors --avg-gap 25 --dataset avg_gap-25 --model net7
python3 source/run_distributed_no_communication.py --controller manual --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 25 --dataset avg_gap-25
python3 source/run_distributed_no_communication.py --controller learned --generate-dataset --plots-dataset --check-dataset --net-input all_sensors --avg-gap 25 --dataset avg_gap-25 --model net7
python3 source/run_distributed_no_communication.py --compare-all --net-input all_sensors --avg-gap 25 --dataset avg_gap-25

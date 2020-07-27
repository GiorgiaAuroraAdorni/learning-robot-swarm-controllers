#!/bin/bash


# Experiment A
#python3 source/task1_extension.py --controller omniscient --generate-dataset --plot-dataset --check-dataset --dataset mixed-1000simulations
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-1000simulations --model net1-extension
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations --model net1-extension

#python3 source/task1_extension.py --model-type communication --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-1000simulations --model net1-extension
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations --model net1-extension

#python3 source/task1.py --model-type communication --compare-all --dataset mixed-1000simulations


# Experiment B
#python3 source/task1_extension.py --controller omniscient --generate-dataset --plot-dataset --check-dataset --dataset mixed-1000simulations
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-long
python3 source/task1_extension.py --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-1000simulations-long --model net1-extension-long
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-long --model net1-extension-long

python3 source/task1_extension.py --model-type communication --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-1000simulations-long --model net1-extension-long
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations --model net1-extension

#python3 source/task1.py --model-type communication --compare-all --dataset mixed-1000simulations


# Experiment C
#python3 source/task1_extension.py --n-simulations 5000 --controller omniscient --generate-dataset --plot-dataset --check-dataset --dataset mixed-5000simulations
#python3 source/task1_extension.py --n-simulations 5000 --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations
#python3 source/task1_extension.py --n-simulations 5000 --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-5000simulations --model net2-extension
#python3 source/task1_extension.py --n-simulations 5000 --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations --model net2-extension
#
#python3 source/task1_extension.py --n-simulations 5000 --model-type communication --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-5000simulations --model net2-extension
#python3 source/task1_extension.py --n-simulations 5000 --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations --model net2-extension
#
#python3 source/task1.py --n-simulations 5000 --model-type communication --compare-all --dataset mixed-5000simulations

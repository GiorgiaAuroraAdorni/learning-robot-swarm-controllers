#!/bin/bash


# Experiment A
#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-1000simulations --model net1-extension
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations --model net1-extension

#python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --save-net --plots-net --dataset mixed-1000simulations --model net1-extension
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations --model net1-extension

#python3 source/task1_extension.py --model-type communication --compare-all --dataset mixed-1000simulations


# Experiment B
#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-long
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-1000simulations-long --model net1-extension-long
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-long --model net1-extension-long

#python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --save-net --plots-net --dataset mixed-1000simulations-long --model net1-extension-long
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-long --model net1-extension-long

#python3 source/task1_extension.py --model-type communication --compare-all --dataset mixed-1000simulations-long


# Experiment C
#python3 source/task1_extension.py --n-simulations 5000 --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations
#python3 source/task1_extension.py --n-simulations 5000 --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations
#python3 source/task1_extension.py --n-simulations 5000 --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-5000simulations --model net2-extension
#python3 source/task1_extension.py --n-simulations 5000 --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations --model net2-extension
#
#python3 source/task1_extension.py --n-simulations 5000 --model-type communication --controller omniscient --train-net --save-net --plots-net --dataset mixed-5000simulations --model net2-extension
#python3 source/task1_extension.py --n-simulations 5000 --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations --model net2-extension

#python3 source/task1_extension.py --n-simulations 5000 --model-type communication --compare-all --dataset mixed-5000simulations


# Experiment D
#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-2
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-2
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-1000simulations-2 --model net1-extension-2
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-2 --model net1-extension-2

#python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --save-net --plots-net --dataset mixed-1000simulations-2 --model net1-extension-2
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-2 --model net1-extension-2

#python3 source/task1_extension.py --model-type communication --compare-all --dataset mixed-1000simulations-2


# Experiment slow omniscient constant 4

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --save-net--plots-net --dataset mixed-1000simulations-slow --model net1-extension-slow
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow --model net1-extension-slow

#python3 source/task1_extension.py --model-type communication --train-net --save-net--controller omniscient --plots-net --dataset mixed-1000simulations-slow --model net1-extension-slow
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow --model net1-extension-slow
#
#python3 source/task1_extension.py --model-type communication --compare-all --dataset mixed-1000simulations-slow


# Experiment very slow omniscient constant 1

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-veryslow
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-veryslow
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --save-net --plots-net --dataset mixed-1000simulations-veryslow --model net1-extension-veryslow
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-veryslow --model net1-extension-veryslow

#python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --save-net --plots-net --dataset mixed-1000simulations-veryslow --model net1-extension-veryslow
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-veryslow --model net1-extension-veryslow
#
#python3 source/task1_extension.py --model-type communication --compare-all --dataset mixed-1000simulations-veryslow


# Experiment slow omniscient constant 4, more timesteps, new net

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow-new
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow-new
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset mixed-1000simulations-slow-new --model net1-extension-slow-new
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow-new --model net1-extension-slow-new

#python3 source/task1_extension.py --model-type communication --train-net --controller omniscient --plots-net --dataset mixed-1000simulations-slow-new --model net1-extension-slow-new
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow-new --model net1-extension-slow-new

#python3 source/task1_extension.py --model-type communication --compare-all --dataset mixed-1000simulations-slow-new

#python3 source/task1_extension.py --generate-animations --dataset mixed-1000simulations-slow-new --model net1-extension-slow-new


# Huge net Experiment slow omniscient constant 4, more timesteps, huge net

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow-huge
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow-huge
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset mixed-1000simulations-slow-huge --model net1-extension-slow-huge
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow-huge --model net1-extension-slow-huge

#python3 source/task1_extension.py --model-type communication --train-net --controller omniscient --plots-net --dataset mixed-1000simulations-slow-huge --model net1-extension-slow-huge
#python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-1000simulations-slow-huge --model net1-extension-slow-huge
#
#python3 source/task1_extension.py --model-type communication --compare-all --dataset mixed-1000simulations-slow-huge

#python3 source/task1_extension.py --generate-animations --dataset mixed-1000simulations-slow-huge --model net1-extension-slow-huge


# Huge net Experiment slow omniscient constant 4, more timesteps, huge net 5000

#python3 source/task1_extension.py --n-simulations 5000 --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations-slow-huge
#python3 source/task1_extension.py --n-simulations 5000 --controller manual --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations-slow-huge
#python3 source/task1_extension.py --n-simulations 5000 --controller omniscient --generate-split --train-net --plots-net --dataset mixed-5000simulations-slow-huge --model net1-extension-5000-slow-huge
#python3 source/task1_extension.py --n-simulations 5000 --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations-slow-huge --model net1-extension-5000-slow-huge

#python3 source/task1_extension.py --n-simulations 5000 --model-type communication --train-net --controller omniscient --plots-net --dataset mixed-5000simulations-slow-huge --model net1-extension-5000-slow-huge
#python3 source/task1_extension.py --n-simulations 5000 --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset mixed-5000simulations-slow-huge --model net1-extension-5000-slow-huge
#
#python3 source/task1_extension.py --n-simulations 5000 --model-type communication --compare-all --dataset mixed-5000simulations-slow-huge

python3 source/task1_extension.py --n-simulations 5000 --generate-animations --dataset mixed-5000simulations-slow-huge --model net1-extension-5000-slow-huge

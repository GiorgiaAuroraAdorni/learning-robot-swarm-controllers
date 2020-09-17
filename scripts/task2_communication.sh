#!/bin/bash

## Generate plots for omniscient datasets

#python3 source/task2.py --dataset avg_gap-8-N-5 --model net-v1 --myt-quantity 5 --avg-gap 8 --controller omniscient --plots-dataset
#python3 source/task2.py --dataset avg_gap-20-N-5 --model net-v2 --avg-gap 20 --myt-quantity 5 --controller omniscient --plots-dataset
#python3 source/task2.py --dataset avg_gap-8-N-8 --model net-v3 --myt-quantity 8 --avg-gap 8 --controller omniscient --plots-dataset
#python3 source/task2.py --dataset avg_gap-20-N-8 --model net-v4 --avg-gap 20 --myt-quantity 8 --controller omniscient --plots-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-5 --model net-v5 --myt-quantity 5 --controller omniscient --plots-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-8 --model net-v6 --myt-quantity 8 --controller omniscient --plots-dataset
#python3 source/task2.py --dataset avg_gap-8-N-variable --model net-v7 --avg-gap 8 --controller omniscient --plots-dataset
#python3 source/task2.py --dataset avg_gap-20-N-variable --model net-v8 --avg-gap 20 --controller omniscient --plots-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-variable --model net-v9 --controller omniscient --plots-dataset


## Generate manual datasets and plots

#python3 source/task2.py --dataset avg_gap-8-N-5 --model net-v1 --myt-quantity 5 --avg-gap 8 --controller manual --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-20-N-5 --model net-v2 --avg-gap 20 --myt-quantity 5 --controller manual --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-8-N-8 --model net-v3 --myt-quantity 8 --avg-gap 8 --controller manual --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-20-N-8 --model net-v4 --avg-gap 20 --myt-quantity 8 --controller manual --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-5 --model net-v5 --myt-quantity 5 --controller manual --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-8 --model net-v6 --myt-quantity 8 --controller manual --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-8-N-variable --model net-v7 --avg-gap 8 --controller manual --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-20-N-variable --model net-v8 --avg-gap 20 --controller manual --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-variable --model net-v9 --controller manual --plots-dataset --generate-dataset


# Train nets and generate plots

#python3 source/task2.py --dataset avg_gap-8-N-5 --model net-v1 --myt-quantity 5 --avg-gap 8 --controller omniscient --plots-net
#python3 source/task2.py --dataset avg_gap-20-N-5 --model net-v2 --avg-gap 20 --myt-quantity 5 --controller omniscient --plots-net
#python3 source/task2.py --dataset avg_gap-8-N-8 --model net-v3 --myt-quantity 8 --avg-gap 8 --controller omniscient --plots-net
#python3 source/task2.py --dataset avg_gap-20-N-8 --model net-v4 --avg-gap 20 --myt-quantity 8 --controller omniscient --plots-net
#python3 source/task2.py --dataset avg_gap-variable-N-5 --model net-v5 --myt-quantity 5 --controller omniscient --plots-net
#python3 source/task2.py --dataset avg_gap-variable-N-8 --model net-v6 --myt-quantity 8 --controller omniscient --plots-net
#python3 source/task2.py --dataset avg_gap-8-N-variable --model net-v7 --avg-gap 8 --controller omniscient --plots-net
#python3 source/task2.py --dataset avg_gap-20-N-variable --model net-v8 --avg-gap 20 --controller omniscient --plots-net
#python3 source/task2.py --dataset avg_gap-variable-N-variable --model net-v9 --controller omniscient --plots-net


# Generate learned datasets and plots

#python3 source/task2.py --dataset avg_gap-8-N-5 --model net-v1 --myt-quantity 5 --avg-gap 8 --controller learned --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-20-N-5 --model net-v2 --avg-gap 20 --myt-quantity 5 --controller learned --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-8-N-8 --model net-v3 --myt-quantity 8 --avg-gap 8 --controller learned --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-20-N-8 --model net-v4 --avg-gap 20 --myt-quantity 8 --controller learned --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-5 --model net-v5 --myt-quantity 5 --controller learned --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-8 --model net-v6 --myt-quantity 8 --controller learned --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-8-N-variable --model net-v7 --avg-gap 8 --controller learned --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-20-N-variable --model net-v8 --avg-gap 20 --controller learned --plots-dataset --generate-dataset
#python3 source/task2.py --dataset avg_gap-variable-N-variable --model net-v9 --controller learned --plots-dataset --generate-dataset

# Generate comparison plots

python3 source/task2.py --dataset avg_gap-8-N-5 --model net-v1 --myt-quantity 5 --avg-gap 8 --controller omniscient --compare-all
python3 source/task2.py --dataset avg_gap-20-N-5 --model net-v2 --avg-gap 20 --myt-quantity 5 --controller omniscient --compare-all
python3 source/task2.py --dataset avg_gap-8-N-8 --model net-v3 --myt-quantity 8 --avg-gap 8 --controller omniscient --compare-all
python3 source/task2.py --dataset avg_gap-20-N-8 --model net-v4 --avg-gap 20 --myt-quantity 8 --controller omniscient --compare-all
python3 source/task2.py --dataset avg_gap-variable-N-5 --model net-v5 --myt-quantity 5 --controller omniscient --compare-all
python3 source/task2.py --dataset avg_gap-variable-N-8 --model net-v6 --myt-quantity 8 --controller omniscient --compare-all
python3 source/task2.py --dataset avg_gap-8-N-variable --model net-v7 --avg-gap 8 --controller omniscient --compare-all
python3 source/task2.py --dataset avg_gap-20-N-variable --model net-v8 --avg-gap 20 --controller omniscient --compare-all
python3 source/task2.py --dataset avg_gap-variable-N-variable --model net-v9 --controller omniscient --compare-all
#!/bin/bash

# ANIMATION + PLOTS
python3 source/task1_extension.py --dataset avg_gap-variable-N-variable --model 18 --generate-animations

# PLOTS datasets
python3 source/task1_extension.py --dataset avg_gap-8-N-5 --model net-d10 --myt-quantity 5 --avg-gap 8 --controller all --plots-dataset
python3 source/task1_extension.py --dataset avg_gap-20-N-5 --model net-d11 --avg-gap 20 --myt-quantity 5 --controller all --plots-dataset
python3 source/task1_extension.py --dataset avg_gap-variable-N-5 --model net-d12 --myt-quantity 5 --controller all --plots-dataset
python3 source/task1_extension.py --dataset avg_gap-8-N-8 --model net-d13 --myt-quantity 8 --avg-gap 8 --controller all --plots-dataset
python3 source/task1_extension.py --dataset avg_gap-20-N-8 --model net-d14 --avg-gap 20 --myt-quantity 8 --controller all --plots-dataset
python3 source/task1_extension.py --dataset avg_gap-variable-N-8 --model net-d15 --myt-quantity 8 --controller all --plots-dataset
python3 source/task1_extension.py --dataset avg_gap-8-N-variable --model net-d16 --avg-gap 8 --controller all --plots-dataset
python3 source/task1_extension.py --dataset avg_gap-20-N-variable --model net-d17 --avg-gap 20 --controller all --plots-dataset
python3 source/task1_extension.py --dataset avg_gap-variable-N-variable --model net-d18 --controller all --plots-dataset

python3 source/task1_extension.py --model-type communication --dataset avg_gap-8-N-5 --model net-c10 --myt-quantity 5 --avg-gap 8 --controller learned_communication --plots-dataset
python3 source/task1_extension.py --model-type communication --dataset avg_gap-20-N-5 --model net-c11 --avg-gap 20 --myt-quantity 5 --controller learned_communication --plots-dataset
python3 source/task1_extension.py --model-type communication --dataset avg_gap-variable-N-5 --model net-c12 --myt-quantity 5 --controller learned_communication --plots-dataset
python3 source/task1_extension.py --model-type communication --dataset avg_gap-8-N-8 --model net-c13 --myt-quantity 8 --avg-gap 8 --controller learned_communication --plots-dataset
python3 source/task1_extension.py --model-type communication --dataset avg_gap-20-N-8 --model net-c14 --avg-gap 20 --myt-quantity 8 --controller learned_communication --plots-dataset
python3 source/task1_extension.py --model-type communication --dataset avg_gap-variable-N-8 --model net-c15 --myt-quantity 8 --controller learned_communication --plots-dataset
python3 source/task1_extension.py --model-type communication --dataset avg_gap-8-N-variable --model net-c16 --avg-gap 8 --controller learned_communication --plots-dataset
python3 source/task1_extension.py --model-type communication --dataset avg_gap-20-N-variable --model net-c17 --avg-gap 20 --controller learned_communication --plots-dataset
python3 source/task1_extension.py --model-type communication --dataset avg_gap-variable-N-variable --model net-c18 --controller learned_communication --plots-dataset

# PLOTS NET
python3 source/task1_extension.py --dataset avg_gap-8-N-5 --model net-d10 --myt-quantity 5 --avg-gap 8 --controller omniscient --plots-net
python3 source/task1_extension.py --dataset avg_gap-20-N-5 --model net-d11 --avg-gap 20 --myt-quantity 5 --controller omniscient --plots-net
python3 source/task1_extension.py --dataset avg_gap-variable-N-5 --model net-d12 --myt-quantity 5 --controller omniscient --plots-net
python3 source/task1_extension.py --dataset avg_gap-8-N-8 --model net-d13 --myt-quantity 8 --avg-gap 8 --controller omniscient --plots-net
python3 source/task1_extension.py --dataset avg_gap-20-N-8 --model net-d14 --avg-gap 20 --myt-quantity 8 --controller omniscient --plots-net
python3 source/task1_extension.py --dataset avg_gap-variable-N-8 --model net-d15 --myt-quantity 8 --controller omniscient --plots-net
python3 source/task1_extension.py --dataset avg_gap-8-N-variable --model net-d16 --avg-gap 8 --controller omniscient --plots-net
python3 source/task1_extension.py --dataset avg_gap-20-N-variable --model net-d17 --avg-gap 20 --controller omniscient --plots-net
python3 source/task1_extension.py --dataset avg_gap-variable-N-variable --model net-d18 --controller omniscient --plots-net

python3 source/task1_extension.py --model-type communication --dataset avg_gap-8-N-5 --model net-c10 --myt-quantity 5 --avg-gap 8 --controller omniscient --plots-net
python3 source/task1_extension.py --model-type communication --dataset avg_gap-20-N-5 --model net-c11 --avg-gap 20 --myt-quantity 5 --controller omniscient --plots-net
python3 source/task1_extension.py --model-type communication --dataset avg_gap-variable-N-5 --model net-c12 --myt-quantity 5 --controller omniscient --plots-net
python3 source/task1_extension.py --model-type communication --dataset avg_gap-8-N-8 --model net-c13 --myt-quantity 8 --avg-gap 8 --controller omniscient --plots-net
python3 source/task1_extension.py --model-type communication --dataset avg_gap-20-N-8 --model net-c14 --avg-gap 20 --myt-quantity 8 --controller omniscient --plots-net
python3 source/task1_extension.py --model-type communication --dataset avg_gap-variable-N-8 --model net-c15 --myt-quantity 8 --controller omniscient --plots-net
python3 source/task1_extension.py --model-type communication --dataset avg_gap-8-N-variable --model net-c16 --avg-gap 8 --controller omniscient --plots-net
python3 source/task1_extension.py --model-type communication --dataset avg_gap-20-N-variable --model net-c17 --avg-gap 20 --controller omniscient --plots-net
python3 source/task1_extension.py --model-type communication --dataset avg_gap-variable-N-variable --model net-c18 --controller omniscient --plots-net

# PLOT COMPARISON
python3 source/task1_extension.py --dataset avg_gap-8-N-5 --model 10 --myt-quantity 5 --avg-gap 8 --compare-all
python3 source/task1_extension.py --dataset avg_gap-20-N-5 --model 11 --avg-gap 20 --myt-quantity 5 --compare-all
python3 source/task1_extension.py --dataset avg_gap-variable-N-5 --model 12 --myt-quantity 5 --compare-all
python3 source/task1_extension.py --dataset avg_gap-8-N-8 --model 13 --myt-quantity 8 --avg-gap 8 --compare-all
python3 source/task1_extension.py --dataset avg_gap-20-N-8 --model 14 --avg-gap 20 --myt-quantity 8 --compare-all
python3 source/task1_extension.py --dataset avg_gap-variable-N-8 --model 15 --myt-quantity 8 --compare-all
python3 source/task1_extension.py --dataset avg_gap-8-N-variable --model 16 --avg-gap 8 --compare-all
python3 source/task1_extension.py --dataset avg_gap-20-N-variable --model 17 --avg-gap 20 --compare-all
python3 source/task1_extension.py --dataset avg_gap-variable-N-variable --model 18 --compare-all


# Fixed avg_gap, fixed N
# avg_gap = 8, N = 5
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-5 --myt-quantity 5 --avg-gap 8
python3 source/task1_extension.py --controller omniscient --generate-split --plots-net --train-net --save-net --dataset avg_gap-8-N-5 --model net-d10 --myt-quantity 5 --avg-gap 8
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-5 --myt-quantity 5 --avg-gap 8
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-5 --model net-d10 --myt-quantity 5 --avg-gap 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-8-N-5 --myt-quantity 5 --avg-gap 8 --model net-c10
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-5 --myt-quantity 5 --avg-gap 8 --model net-c10

# Fixed avg_gap, fixed N
# avg_gap = 20, N = 5
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-5 --avg-gap 20 --myt-quantity 5
python3 source/task1_extension.py --controller omniscient --generate-split --plots-net --train-net --save-net --dataset avg_gap-20-N-5 --model net-d11 --avg-gap 20 --myt-quantity 5
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-5 --avg-gap 20 --myt-quantity 5
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-5 --model net-d11 --avg-gap 20 --myt-quantity 5

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-20-N-5 --avg-gap 20 --myt-quantity 5 --model net-c11
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-5 --avg-gap 20 --myt-quantity 5 --model net-c11


# Variable avg_gap, fixed N
# avg_gap = [5, 25], N = 5
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-5 --myt-quantity 5
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-5 --myt-quantity 5
python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-variable-N-5 --model net-d12 --myt-quantity 5
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-5 --model net-d12 --myt-quantity 5

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-variable-N-5 --myt-quantity 5 --model net-c12
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-5 --myt-quantity 5 --model net-c12


# Fixed avg_gap, fixed N
# avg_gap = 8, N = 8
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-8 --myt-quantity 8 --avg-gap 8
python3 source/task1_extension.py --controller omniscient --generate-split --plots-net --train-net --save-net --dataset avg_gap-8-N-8 --model net-d13 --myt-quantity 8 --avg-gap 8
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-8 --myt-quantity 8 --avg-gap 8
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-8 --model net-d13 --myt-quantity 8 --avg-gap 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-8-N-8 --myt-quantity 8 --avg-gap 8 --model net-c13
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-8 --myt-quantity 8 --avg-gap 8 --model net-c13


# Fixed avg_gap, fixed N
# avg_gap = 20, N = 8
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-8 --avg-gap 20 --myt-quantity 8
python3 source/task1_extension.py --controller omniscient --generate-split --plots-net --train-net --save-net --dataset avg_gap-20-N-8 --model net-d14 --avg-gap 20 --myt-quantity 8
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-8 --avg-gap 20 --myt-quantity 8
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-8 --model net-d14 --avg-gap 20 --myt-quantity 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-20-N-8 --avg-gap 20 --myt-quantity 8 --model net-c14
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-8 --avg-gap 20 --myt-quantity 8 --model net-c14



# Variable avg_gap, fixed N
# avg_gap = [5, 25], N = 8
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-8 --myt-quantity 8
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-8 --myt-quantity 8
python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-variable-N-8 --model net-d15 --myt-quantity 8
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-8 --model net-d15 --myt-quantity 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-variable-N-8 --myt-quantity 8 --model net-c15
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-8 --myt-quantity 8 --model net-c15


# Fixed avg_gap, variable N
# avg_gap = 8, N = [5, 10]
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-variable --avg-gap 8
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-variable --avg-gap 8
python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-8-N-variable --model net-d16 --avg-gap 8
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-variable --model net-d16 --avg-gap 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-8-N-variable --avg-gap 8 --model net-c16
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-variable --avg-gap 8 --model net-c16


# Fixed avg_gap, variable N
# avg_gap = 20, N = [5, 10]
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-variable --avg-gap 20
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-variable --avg-gap 20
python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-20-N-variable --model net-d17 --avg-gap 20
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-variable --model net-d17 --avg-gap 20

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-20-N-variable --avg-gap 20 --model net-c17
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-variable --avg-gap 20 --model net-c17


# Variable avg_gap, variable N
# avg_gap = [5, 25], N = [5, 10]
python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-variable
python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-variable
python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-variable-N-variable --model net-d18
python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-variable --model net-d18

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-variable-N-variable --model net-c18
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-variable --model net-c18

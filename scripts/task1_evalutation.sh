# PLOTS compare all

#python3 source/task1_extension.py --dataset avg_gap-8-N-5 --model net-v1 --myt-quantity 5 --avg-gap 8 --compare-all
#python3 source/task1_extension.py --dataset avg_gap-20-N-5 --model net-v2 --avg-gap 20 --myt-quantity 5 --compare-all
#python3 source/task1_extension.py --dataset avg_gap-8-N-8 --model net-v3 --myt-quantity 8 --avg-gap 8 --compare-all
#python3 source/task1_extension.py --dataset avg_gap-20-N-8 --model net-v4 --avg-gap 20 --myt-quantity 8 --compare-all
#python3 source/task1_extension.py --dataset avg_gap-variable-N-5 --model net-v5 --myt-quantity 5 --compare-all
#python3 source/task1_extension.py --dataset avg_gap-variable-N-8 --model net-v6 --myt-quantity 8 --compare-all
#python3 source/task1_extension.py --dataset avg_gap-8-N-variable --model net-v7 --avg-gap 8 --compare-all
#python3 source/task1_extension.py --dataset avg_gap-20-N-variable --model net-v8 --avg-gap 20 --compare-all
#python3 source/task1_extension.py --dataset avg_gap-variable-N-variable --model net-v9 --compare-all


# Fixed avg_gap, fixed N
# avg_gap = 8, N = 5

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-5 --myt-quantity 5 --avg-gap 8
#python3 source/task1_extension.py --controller omniscient --generate-split --plots-net --train-net --save-net --dataset avg_gap-8-N-5 --model net-v1 --myt-quantity 5 --avg-gap 8
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-5 --myt-quantity 5 --avg-gap 8
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-5 --model net-v1 --myt-quantity 5 --avg-gap 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-8-N-5 --myt-quantity 5 --avg-gap 8
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-5 --myt-quantity 5 --avg-gap 8

# Fixed avg_gap, fixed N
# avg_gap = 20, N = 5

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-5 --avg-gap 20 --myt-quantity 5
#python3 source/task1_extension.py --controller omniscient --generate-split --plots-net --train-net --save-net --dataset avg_gap-20-N-5 --model net-v2 --avg-gap 20 --myt-quantity 5
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-5 --avg-gap 20 --myt-quantity 5
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-5 --model net-v2 --avg-gap 20 --myt-quantity 5

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-20-N-5 --avg-gap 20 --myt-quantity 5
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-5 --avg-gap 20 --myt-quantity 5


# Fixed avg_gap, fixed N
# avg_gap = 8, N = 8

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-8 --myt-quantity 8 --avg-gap 8
#python3 source/task1_extension.py --controller omniscient --generate-split --plots-net --train-net --save-net --dataset avg_gap-8-N-8 --model net-v3 --myt-quantity 8 --avg-gap 8
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-8 --myt-quantity 8 --avg-gap 8
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-8 --model net-v3 --myt-quantity 8 --avg-gap 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-8-N-8 --myt-quantity 8 --avg-gap 8
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-8 --myt-quantity 8 --avg-gap 8


# Fixed avg_gap, fixed N
# avg_gap = 20, N = 8

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-8 --avg-gap 20 --myt-quantity 8
#python3 source/task1_extension.py --controller omniscient --generate-split --plots-net --train-net --save-net --dataset avg_gap-20-N-8 --model net-v4 --avg-gap 20 --myt-quantity 8
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-8 --avg-gap 20 --myt-quantity 8
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-8 --model net-v4 --avg-gap 20 --myt-quantity 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-20-N-8 --avg-gap 20 --myt-quantity 8
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-8 --avg-gap 20 --myt-quantity 8


# Variable avg_gap, fixed N
# avg_gap = [5, 25], N = 5

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-5 --myt-quantity 5
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-5 --myt-quantity 5
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-variable-N-5 --model net-v5 --myt-quantity 5
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-5 --model net-v5 --myt-quantity 5

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-variable-N-5 --myt-quantity 5
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-5 --myt-quantity 5


# Variable avg_gap, fixed N
# avg_gap = [5, 25], N = 8

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-8 --myt-quantity 8
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-8 --myt-quantity 8
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-variable-N-8 --model net-v6 --myt-quantity 8
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-8 --model net-v6 --myt-quantity 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-variable-N-8 --myt-quantity 8
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-8 --myt-quantity 8


# Fixed avg_gap, variable N
# avg_gap = 8, N = [5, 10]

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-variable --avg-gap 8
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-variable --avg-gap 8
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-8-N-variable --model net-v7 --avg-gap 8
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-variable --model net-v7 --avg-gap 8

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-8-N-variable --avg-gap 8
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-8-N-variable --avg-gap 8


# Fixed avg_gap, variable N
# avg_gap = 20, N = [5, 10]

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-variable --avg-gap 20
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-variable --avg-gap 20
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-20-N-variable --model net-v8 --avg-gap 20
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-variable --model net-v8 --avg-gap 20

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-20-N-variable --avg-gap 20
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-20-N-variable --avg-gap 20


# Variable avg_gap, variable N
# avg_gap = [5, 25], N = [5, 10]

#python3 source/task1_extension.py --controller omniscient --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-variable
#python3 source/task1_extension.py --controller manual --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-variable
#python3 source/task1_extension.py --controller omniscient --generate-split --train-net --plots-net --dataset avg_gap-variable-N-variable --model net-v9
#python3 source/task1_extension.py --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-variable --model net-v9

python3 source/task1_extension.py --model-type communication --controller omniscient --train-net --plots-net --dataset avg_gap-variable-N-variable
python3 source/task1_extension.py --model-type communication --controller learned --generate-dataset --plots-dataset --check-dataset --dataset avg_gap-variable-N-variable
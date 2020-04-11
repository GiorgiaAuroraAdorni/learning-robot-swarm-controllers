import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from utils import check_dir

out_dir = 'models/distributed/'
check_dir(out_dir)
file_name = os.path.join(out_dir, 'dataset_split.npy')

# Uncomment the following line to generate a new dataset split
# dataset_split(file_name)

x = np.load(file_name)  # load
train_indices, test_indices = x[:800], x[800:]

# Split the datasets also defining input and output
runs_dir = 'out-5myts/'

train_sample = []
test_sample = []

train_target = []
test_target = []

for file_name in os.listdir(runs_dir):
    file = os.path.join(runs_dir, file_name)
    run = pd.read_pickle(file)

    i = int(re.findall('\d+', file_name)[0])

    if i in train_indices:
        input = train_sample
        output = train_target
    else:
        input = test_sample
        output = test_target

    for step in run:
        for myt in step:
            # The input is the prox_values, that are the response values of ​​the sensors [array of 7 floats]
            sample = myt['prox_values'].copy()
            input.append(sample)

            # The output is the speed of the wheels (which we assume equals left and right) [array of 1 float]
            speed = myt['motor_left_target']
            output.append([speed])

# Generate the tensors
train_sample_tensor = torch.FloatTensor(train_sample)
test_sample_tensor = torch.FloatTensor(test_sample)

train_target_tensor = torch.FloatTensor(train_target)
test_target_tensor = torch.FloatTensor(test_target)


print()

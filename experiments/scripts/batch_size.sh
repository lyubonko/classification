#!/bin/bash

# this series to compare different optimizers

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --DATASET_batch_size 32
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --DATASET_batch_size 64
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --DATASET_batch_size 128
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --DATASET_batch_size 256

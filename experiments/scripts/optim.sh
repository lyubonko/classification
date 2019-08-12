#!/bin/bash

# this series to compare different optimizers

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_optim sgd
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_optim adam
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_optim rms_prop

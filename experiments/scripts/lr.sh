#!/bin/bash

# this series can be used to determine appropriate learning rate

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_lr 0.05   --TRAIN_epochs 10
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_lr 0.01   --TRAIN_epochs 10
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_lr 0.005  --TRAIN_epochs 10
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_lr 0.001  --TRAIN_epochs 10
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_lr 0.0005 --TRAIN_epochs 10
python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml --TRAIN_lr 0.0001 --TRAIN_epochs 10
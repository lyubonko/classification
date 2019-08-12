#!/bin/bash

# run on tiny dataset to overfit data

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml \
                         --DATASET_batch_size 100 --DATASET_tiny True \
                         --DATASET_transforms_train init --DATASET_transforms_val init \
                         --TRAIN_lr 0.01   --TRAIN_epochs 300

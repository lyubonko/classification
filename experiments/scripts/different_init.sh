#!/bin/bash

# different init strategy

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml \
                         --TRAIN_epochs 5 \
                         --MODEL_name vgg_in3x32x32_out10 \
                         --MODEL_init default

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml \
                         --TRAIN_epochs 30 \
                         --MODEL_name vgg_in3x32x32_out10 \
                         --MODEL_init resnet

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml \
                         --TRAIN_epochs 30 \
                         --MODEL_name vgg_bn_in3x32x32_out10 \
                         --MODEL_init default

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml \
                         --TRAIN_epochs 30 \
                         --MODEL_name vgg_bn_in3x32x32_out10 \
                         --MODEL_init resnet

python src/main_train.py --param_file experiments/cfgs/params_cifar10.yaml \
                         --TRAIN_epochs 50 \
                         --MODEL_name vgg_bn_in3x32x32_out10 \
                         --MODEL_init resnet \
                         --DATASET_transforms_train augment
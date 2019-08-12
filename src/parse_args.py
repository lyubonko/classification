from argparse import ArgumentParser

import yaml


def parse_arguments():

    parser = ArgumentParser()
    parser.add_argument('--param_file', type=str, help='configure file with parameters')
    parser.add_argument('--TRAIN_lr', default=None, type=float, help='initial learning rate')
    parser.add_argument('--TRAIN_weight_decay', default=None, type=float)
    parser.add_argument('--TRAIN_epochs', default=None, type=int, help='number of training epochs')
    parser.add_argument('--TRAIN_optim', default=None, type=str, help='training optimizer')
    parser.add_argument('--DATASET_batch_size', default=None, type=int, help='train batch size')
    parser.add_argument('--DATASET_tiny', default=None, type=bool, help='train on tiny subset') # TODO : bool in args
    parser.add_argument('--DATASET_transforms_train', default=None, type=str, help='train transform')
    parser.add_argument('--DATASET_transforms_val', default=None, type=str, help='val transform')
    parser.add_argument('--MODEL_name', default=None, type=str)
    parser.add_argument('--MODEL_init', default=None, type=str)
    parser.add_argument('--MODEL_weights', default=None, type=str)
    args = parser.parse_args()

    # parse param file
    with open(args.param_file, 'r') as f:
        params = yaml.safe_load(f)

    # addition configurations
    if args.TRAIN_weight_decay is not None:
        params['TRAIN']['weight_decay'] = args.TRAIN_weight_decay

    if args.TRAIN_lr is not None:
        params['TRAIN']['lr'] = args.TRAIN_lr

    if args.TRAIN_epochs is not None:
        params['TRAIN']['epochs'] = args.TRAIN_epochs

    if args.TRAIN_optim is not None:
        params['TRAIN']['optim'] = args.TRAIN_optim

    if args.DATASET_batch_size is not None:
        params['DATASET']['batch_size'] = args.DATASET_batch_size

    if args.DATASET_tiny is not None:
        params['DATASET']['tiny'] = args.DATASET_tiny

    if args.DATASET_transforms_val is not None:
        params['DATASET']['transforms']['val'] = args.DATASET_transforms_val

    if args.DATASET_transforms_train is not None:
        params['DATASET']['transforms']['train'] = args.DATASET_transforms_train

    if args.MODEL_name is not None:
        params['MODEL']['name'] = args.MODEL_name

    if args.MODEL_init is not None:
        params['MODEL']['init'] = args.MODEL_init

    if args.MODEL_weights is not None:
        params['MODEL']['weights'] = args.MODEL_weights

    return params

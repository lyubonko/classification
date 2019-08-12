import os
import warnings

import numpy as np
import torch


def handle_device(with_cuda):
    """
    :param with_cuda: flag to use cuda device (GPU) or not
    :return: device specification
    """

    device = 'cpu'
    cuda_available = torch.cuda.is_available()
    if cuda_available and with_cuda:
        device = 'cuda'
    elif cuda_available and not with_cuda:
        warnings.warn("WARNING: parameter 'with_cuda' is equal 'False', but GPU device is available.\n"
                      "         calculation is done on CPU")
    if not cuda_available and with_cuda:
        warnings.warn("WARNING: parameter 'with_cuda' is equal 'True', but GPU device is unavailable.\n"
                      "         calculation is done on CPU")

    return device


def load_checkpoint(log, model, checkpoint_file, optimizer):

    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        best_prec = checkpoint['best_prec']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        log.log_global("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, start_epoch))
        return start_epoch, best_prec
    else:
        raise ValueError("load_checkpoint: The checkpoint: " + checkpoint_file + " does not exist.")


def save_checkpoint(state, model, params, is_best):
    if is_best:
        # model
        save_name = os.path.join(params['path_save'],
                                 params['experiment_name'] + "_model_best.pth")
        torch.save(model.to('cpu').state_dict(), save_name)
        model.to(params['device'])

        # checkpoint
        save_name = os.path.join(params['path_save'],
                                 params['experiment_name'] +
                                 "_checkpoint_best.pth.tar")

        torch.save(state, save_name)


def next_expr_name(path_dir, start_pattern, n_digit_length):
    len_start = len(start_pattern)
    len_end = len_start + n_digit_length
    present_numbers = [int(x[len_start:len_end]) for x in os.listdir(path_dir) if x.startswith(start_pattern)]
    next_number = np.max([0] + present_numbers) + 1
    return start_pattern + str(next_number).zfill(n_digit_length)

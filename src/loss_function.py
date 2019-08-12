import torch.nn as nn


def set_loss_function(params):

    return nn.CrossEntropyLoss().to(params['device'])

import torch
import torch.utils.data as data
import torchvision.datasets as dsets

import numpy as np

from datasets.transforms_cifar10 import *


class DataSetCifar10(object):
    """
    Class manage CIFAR10 data-set
    """

    def __init__(self,
                 path_data,
                 num_dunkeys=4,
                 batch_size_train=100,
                 batch_size_val=100,
                 download=False,
                 tiny=False,
                 transform_keys=None):

        if transform_keys is None:
            transform_keys = {'train': "init",
                              'val': "init"}

        self.batch_sizes = {'train': batch_size_train, 'val': batch_size_val}

        self.transforms = {'train': transforms_c10[transform_keys['train']],
                           'val': transforms_c10[transform_keys['val']]}

        self.dataset = {}
        self.loader = {}
        for t in ['train', 'val']:
            self.dataset[t] = dsets.CIFAR10(root=path_data,
                                            train=(t == 'train'),
                                            download=download,
                                            transform=self.transforms[t])
            self.loader[t] = torch.utils.data.DataLoader(dataset=self.dataset[t],
                                                         batch_size=self.batch_sizes[t],
                                                         shuffle=(t == 'train'),
                                                         num_workers=num_dunkeys)

        if tiny:
            tiny_trainset = torch.utils.data.dataset.Subset(self.dataset['train'], np.arange(self.batch_sizes['train']))
            tiny_loader = torch.utils.data.DataLoader(tiny_trainset, batch_size=self.batch_sizes['train'])
            for t in ['train', 'val']:
                self.dataset[t] = tiny_trainset
                self.loader[t] = tiny_loader

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn


def init_net(net, init_type):
    if init_type == 'resnet':
        weights_init_resnet(net)
    elif init_type == 'xavier':
        weights_init_xavier(net)

# Official documentation
# source: https://pytorch.org/docs/stable/_modules/torch/nn/init.html

# Example of possible 'Initializer'
# source: https://github.com/3ammor/Weights-Initializer-pytorch/blob/master/weight_initializer.py

# ResNet-like initialization
# source: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L112-L118
def weights_init_resnet(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# Xavier-like init
# source: proposed in pytorch forum
def weights_init_xavier(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# Default for Conv
# source: http://pytorch.org/docs/master/nn.html?highlight=init#torch-nn-init
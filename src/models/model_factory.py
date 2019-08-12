from models.lenet_in1x28x28_out10 import LeNet1x28x28
from models.lenet_in3x32x32_out10 import LeNet3x32x32
from models.lenet_in3x32x32_out47 import LeNet3x32x32out47
from models.vgg_in3x32x32_out10 import Vgg3x32x32
from models.vgg_bn_in3x32x32_out10 import VggBb3x32x32
from models.vgg_bn_pre_in3x32x32_out10 import VggBnPre3x32x32

import torch
from models.init_net import init_net


class ModelFactory(object):
    """
    Model simple factory method
    """

    @staticmethod
    def create(params):
        """
        Creates Model based on detector type
        :param params: Model settings
        :return: Model instance. In case of unknown Model type throws exception.
        """

        if params['MODEL']['name'] == 'lenet_in1x28x28_out10':
            net = LeNet1x28x28()
        elif params['MODEL']['name'] == 'lenet_in3x32x32_out10':
            net = LeNet3x32x32()
        elif params['MODEL']['name'] == 'lenet_in3x32x32_out47':
            net = LeNet3x32x32out47()
        elif params['MODEL']['name'] == 'vgg_in3x32x32_out10':
            net = Vgg3x32x32()
        elif params['MODEL']['name'] == 'vgg_bn_in3x32x32_out10':
            net = VggBb3x32x32()
        elif params['MODEL']['name'] == 'vgg_bn_pre_in3x32x32_out10':
            net = VggBnPre3x32x32()
        else:
            raise ValueError("ModelFactory(): Unknown Model type: " + params['Model']['type'])

        if len(params['MODEL']['weights']) > 0:
            net.load_state_dict(torch.load(params['MODEL']['weights']))
        else:
            init_net(net, params['MODEL']['init'])

        net = net.to(params['device'])

        return net

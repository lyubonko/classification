from collections import OrderedDict
import torch
import torch.nn as nn


def summary(input_size, model):
    """
    Source: https://github.com/warmspringwinds/pytorch-segmentation-detection
    :param input_size:
    :param model:
    :return:
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if not isinstance(module, nn.Sequential) and \
                not isinstance(module, nn.ModuleList) and \
                not (module == model):
            hooks.append(module.register_forward_hook(hook))

    dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(1, *in_size).type(dtype) for in_size in input_size]
    else:
        x = torch.rand(1, *input_size).type(dtype)

    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print('----------------------------------------------------------------')
    line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
    print(line_new)
    print('================================================================')
    line_new = '{:>20}  {:>25} {:>15}'.format('input', str(list(x.size())), '0')
    print(line_new)
    total_params = 0
    trainable_params = 0
    for layer in summary:
        ## input_shape, output_shape, trainable, nb_params
        line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']),
                                                  summary[layer]['nb_params'])
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)
    print('================================================================')
    print('Total params: ' + str(total_params.item()))
    print('Trainable params: ' + str(trainable_params.item()))
    print('Non-trainable params: ' + str(total_params.item() - trainable_params.item()))
    print('----------------------------------------------------------------')
    return summary

from datasets.ds_mnist import DataSetMnist
from datasets.ds_cifar10 import DataSetCifar10
from datasets.ds_dtd import DataSetDTD


class DatasetFactory(object):
    """
    Dataset simple factory method
    """

    @staticmethod
    def create(params):
        """
        Creates Dataset based on detector type
        :param params: Dataset settings
        :return: Dataset instance. In case of unknown Dataset type throws exception.
        """

        if params['DATASET']['name'] == 'mnist':
            return DataSetMnist(params['DATASET']['path'],
                                batch_size_train=params['DATASET']['batch_size'],
                                batch_size_val=params['DATASET']['batch_size_val'],
                                download=params['DATASET']['download'],
                                tiny=params['DATASET']['tiny'],
                                transform_keys=params['DATASET']['transforms'])
        elif params['DATASET']['name'] == 'cifar10':
            return DataSetCifar10(params['DATASET']['path'],
                                  batch_size_train=params['DATASET']['batch_size'],
                                  batch_size_val=params['DATASET']['batch_size_val'],
                                  download=params['DATASET']['download'],
                                  tiny=params['DATASET']['tiny'],
                                  transform_keys=params['DATASET']['transforms'])
        elif params['DATASET']['name'] == 'dtd':
            return DataSetDTD(params['DATASET']['path'],
                              batch_size_train=params['DATASET']['batch_size'],
                              batch_size_val=params['DATASET']['batch_size_val'],
                              tiny=params['DATASET']['tiny'],
                              transform_keys=params['DATASET']['transforms'])

        raise ValueError("DatasetFactory(): Unknown Dataset type: " + params['Dataset']['type'])

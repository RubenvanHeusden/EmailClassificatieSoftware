"""
This file contains the implementation of the CustomDataLoader class which is used to convert the CSVdataset
into batches for word embedding indices that can be fed to the BiLSTM and CNN neural network models.
"""

import torch
from code_utils.dataiterator import DataIterator
from torchtext.data import BucketIterator, Iterator, Dataset


class CustomDataLoader:
    """
    This class implements a CustomDataLoader that is used to convert the contents of a CSV dataset into
    the right format for the BiLSTM and CNN neural networks, which includes converting the words to indices
    in the word embedding matrix uses by the models and converting the labels of the data points into unique
    integers

    Attributes
    ---------
    data: (sub) class of torch.data.Dataset containing the read-in data from a csv file.

    """
    def __init__(self, data: Dataset):
        """
        :param data: (sub) class of torch.data.Dataset containing the read-in data from a csv file.
        """
        self.data = data

    def construct_iterators(self, batch_size: int, is_test_set: bool = False,text_col_name: str = 'text',
                            label_col_name: str = 'label') -> DataIterator:
        """
        This method is used to construct iterators that can be used during the training of the neural
        networks from the dataset created by CSVDataset. When used for evaluation the 'is_test_set' should
        be set to True.

        :param batch_size: integer specifying the size of the batches used in the training of a model
        :param is_test_set: Boolean specifying whether or not the iterator is used as a test set or not.
        When set to True this disables the shuffling of the dataset
        :param text_col_name: string specifying the name of the text attribute in the iterator class
        :param label_col_name: string specifying the name of the label attribute in the iterator class
        :return: Iterator that can be used by the CNN and BiLSTM methods
        """
        self.data.fields[text_col_name].build_vocab(self.data)
        self.data.fields[label_col_name].build_vocab(self.data)
        if not is_test_set:
            return DataIterator(BucketIterator(self.data, batch_size=batch_size, device=torch.device("cpu"),
                                               sort_within_batch=False, sort_key=lambda a: len(a.text), repeat=False),
                                text_col_name, label_col_name)
        else:
            return DataIterator(Iterator(self.data, batch_size=batch_size, device=torch.device("cpu"), sort=False,
                                         sort_within_batch=False, repeat=False, shuffle=False), text_col_name,
                                label_col_name)

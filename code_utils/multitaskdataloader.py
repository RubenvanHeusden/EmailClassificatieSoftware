from torchtext.data import BucketIterator, Iterator
from code_utils.multitaskdataiterator import MultitaskDataIterator
"""
This class implements a dataloader class designed for loading
in a dataset that contains multiple distinct labels for each data point
(for example, intent and emotion)

The datapoints are loaded in separately with one label each time and an appropriate task
associated with it

This takes the field names in the right order as input, so that the dataloader
can return tuples of the right shape
"""


class MultitaskCustomDataLoader:
    def __init__(self, data, text_field, field_names):
        """
        :param text_field: Object of the torchtext.data.Field type used to load in the text \
        from the dataset
        """
        self.data = data
        self.text_field = text_field
        self.field_names = field_names

    def construct_iterators(self, batch_size: int, device, is_test_set: bool = False):
        """
        (see https://torchtext.readthedocs.io/en/latest/vocab.html for possible options)
        :param batch_size: integer specifying the size of the batches
        :param device: torch.Device indicating whether to run on CPU / GPU
        :return: list containing the iterators for train, eval and (test)
        """
        iterators = []
        # Build the vocabulary for the data, this converts all the words into integers
        # pointing to the corresponding rows in the word embedding matrix
        self.text_field.build_vocab(self.data)
        for key, val in self.data.fields.items():
            if key != "text" and key != 'id' and val:
                val.build_vocab(self.data)
        # Construct an iterator specifically for training
        if not is_test_set:
            return MultitaskDataIterator(BucketIterator(
                self.data,
                batch_size=batch_size,
                device=device,
                sort_within_batch=False,
                sort_key=lambda a: len(a.text),
                repeat=False), label_name=self.field_names)

        else:
            return MultitaskDataIterator(Iterator(self.data, batch_size=batch_size,
                         device=device, sort=False,
                         sort_within_batch=False,
                         repeat=False,
                         shuffle=False), label_name=self.field_names)





"""
This file contains the implementation of the DataIterator wrapper class that is used to wrap the Iterators
returned by the torchtext package so that the attributes can be extracted more easily
"""


class DataIterator:
    """
    This class implements a wrapper around a torchtext Iterator class, automatically retrieving the attributes
    of the iterator by name and allowing for a more intuitive way of iterating over the dataset

    Attributes
    ----------
    iterator
        (sub)class of the torchtext.Iterator class containing the data

    text_col_name: str
        string specifying the name of the text attribute in the iterator class

    label_col_name: str
        string specifying the name of the label attribute in the iterator class
    """

    def __init__(self, iterator, text_col_name: str = "text", label_col_name: str = "label"):
        """
        :param iterator: (sub)class of the torchtext.Iterator class containing the data
        :param text_col_name: string specifying the name of the text attribute in the iterator class
        :param label_col_name: string specifying the name of the label attribute in the iterator class
        """
        self.iterator = iterator
        self.text_col_name = text_col_name
        self.label_col_name = label_col_name

    def __iter__(self) -> tuple:
        """
        this method implements a wrapper around the torchtext dataset that retrieves the attributes in the
        torchtext iterator and returns the actual data
        :return: tuple with contents [text_data, label_data] gotten from the torchtext iterator
        """
        for batch in self.iterator:
            yield getattr(batch, self.text_col_name), getattr(batch, self.label_col_name)

    def __len__(self) -> int:
        return len(self.iterator)

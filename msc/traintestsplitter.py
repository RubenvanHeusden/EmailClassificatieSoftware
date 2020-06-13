"""
This file contains an implementation of a class that can be used to split a dataset contained in a single
file into separate train and test files which are used for the training procedures of the models.
It cantains several often used ways of splitting the dataset. If the dataset is already divided into
train and test sets this procedure is not necessary.
"""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedShuffleSplit


class TrainTestSplitter:
    """
    This class implements several functions to stratify a dataset, either by creating splits from
    one datafile are by combining and reshuffling the existing train and test file


    Methods
    -------
    stratify_file(file_path, delimiter, quotechar, text_col_name, label_col_name)
        Method that can be used to read in an existing file and split it into
        train and test files with a stratification strategy to keep the distributions of
        labels between the two files as similar as possible.

    _stratify(dataframe, text_col_name, label_col_name)
        Method that handles the stratificaton of a dataframe and splits it into two parts,
        this method is used by the 'stratify_file' and the 'reshuffle' methods under the hood.

    reshuffle(train_file_path, test_file_path, delimiter, quotechar, text_col_name, label_col_name, shuffle_strategy)
        This methods reads in existing train and test files and reshuffles them so that they are stratified.
    """
    def __init__(self):
        pass

    @staticmethod
    def stratify_file(file_path: str, delimiter: str = ",", quotechar: str = '"', text_col_name: str = "text",
                      label_col_name: str = 'label', train_portion: int = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param file_path: string specifying the path where the datafile is located

        :param delimiter: string specifying the delimiter used in the csv file, this is used by the csv
        reader

        :param quotechar: string indicating the character used for quoting

        :param text_col_name: string specifying the name of the header of the text column, defaults to 'text'

        :param label_col_name: string specifying the name of the header of the label column, defaults to 'label'

        :param train_portion: integer indicating the percentage of the total dataset that is used for training
        the test set size is set to (1-train_portoin), default is 0.7.

        :return: returns a tuple of two dataframes wher the first is the train dataframe and the second is the test
        dataframe
        """

        assert(0.0 < train_portion < 1.0)

        dataframe = pd.read_csv(file_path, sep=delimiter, quotechar=quotechar)
        shuffler = StratifiedShuffleSplit(n_splits=1, train_size=train_portion, test_size=(1-train_portion))
        train_indices, test_indices = list(shuffler.split(dataframe[text_col_name], dataframe[label_col_name]))[0]
        return dataframe.take(train_indices), dataframe.take(test_indices)

    @staticmethod
    def _stratify(dataframe: pd.DataFrame, text_col_name: str, label_col_name: str,
                  train_portion: int = 0.7) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param dataframe: dataframe to be split containing all the datapoints

        :param text_col_name: string specifying the name of the header of the text column, defaults to 'text'

        :param label_col_name: string specifying the name of the header of the label column, defaults to 'label'

        :param train_portion: integer indicating the percentage of the total dataset that is used for training
        the test set size is set to (1-train_portoin), default is 0.7.

        :return: returns a tuple of two dataframes wher the first is the train dataframe and the second is the test
        dataframe
        """

        assert(0.0 < train_portion < 1.0)

        shuffler = StratifiedShuffleSplit(n_splits=1, train_size=train_portion, test_size=(1-train_portion))
        train_indices, test_indices = list(shuffler.split(dataframe[text_col_name], dataframe[label_col_name]))[0]
        return dataframe.take(train_indices), dataframe.take(test_indices)

    def reshuffle(self, train_file_path: str, test_file_path: str, delimiter: str = ",", quotechar: str = '"',
                  text_col_name: str = 'text', label_col_name: str = "label",
                  train_portion: int = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This method can be used to 'reshuffle' existing train and test files when they already exist
        but a different distribution is required. This can be useful when the distribution of labels in one
        of the two files is significantly different from that of the other file.

        :param train_file_path: string specifying the the name of the file where the training set is located

        :param test_file_path: string specifying the name of the file where the test set is located

        :param delimiter: string specifying the delimiter used in the csv file, this is used by the csv
        reader

        :param quotechar: string indicating the character used for quoting

        :param text_col_name: string specifying the name of the header of the text column, defaults to 'text'

        :param label_col_name: string specifying the name of the header of the label column, defaults to 'label'

        :param train_portion: integer indicating the percentage of the total dataset that is used for training
        the test set size is set to (1-train_portoin), default is 0.7.

        :return: returns a tuple of two dataframes wher the first is the train dataframe and the second is the test
        dataframe
        """

        assert(0.0 < train_portion < 1.0)

        train_dataframe = pd.read_csv(train_file_path, quotechar=quotechar, sep=delimiter)
        test_dataframe = pd.read_csv(test_file_path, quotechar=quotechar, sep=delimiter)
        merged_dataframe = pd.concat([train_dataframe, test_dataframe])
        train_frame, test_frame = self._stratify(merged_dataframe, label_col_name=label_col_name,
                                                 text_col_name=text_col_name, train_portion=train_portion)
        return train_frame, test_frame

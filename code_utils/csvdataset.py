"""
This file contains the implementation of the CSVDataset that can be used to read data from a csv file
and convert it into a format suited for further processing and usage by the BiLSTM and CNN models in this
package
"""

import torch
import pandas as pd
from torchtext.data import TabularDataset, LabelField, Field


class CSVDataset:
    """
    This class implements a csv reader that converts the data into an appropriate representation that is used
    further in the pipeline by the biLSTM and CNN neural network models

    Attributes
    ----------
    text_field: Field
        orch.data.Field class set with the appropriate arguments used for reading in and
        (pre) processing the data. (see https://torchtext.readthedocs.io/en/latest/data.html) for more information

    file_name:string
        string specifying the name and location of the csv file containing the training data

    """

    def __init__(self, text_field: Field, file_name: str):
        """
        :param text_field: torch.data.Field class set with the appropriate arguments used for reading in and \
        (pre) processing the data. (see https://torchtext.readthedocs.io/en/latest/data.html) for more information
        :param file_name: string specifying the name and location of the csv file containing the training data
        """

        self.text_field = text_field
        self.file_name = file_name

    def load(self, delimiter: str = ",", quotechar: str = '"', text_col_name: str = 'text',
             label_col_name: str = 'label') -> TabularDataset:
        """

        This methods is responsible for loading in the data from the csv file and converting it into
        a torchtext TabularDataset, it will automatically only select the columns from the file that are
        specified by the 'text_col_name' and 'label_col_name' parameters

        :param delimiter: string specifying the delimiter used when reading in the csv file
        :param quotechar: string specifying the quotechar used when reading in the csvfile
        :param text_col_name: string specifying the name of the column in the csv file containing \
        the text of the data point
        :param label_col_name: string specifying the name of the column in the csv file containing the \
        label of the datapoint
        :return: torch.data.TabularDataset
        """
        file_headers = list(pd.read_csv(self.file_name, sep=delimiter, quotechar=quotechar))
        dset_row = []
        for header in file_headers:
            if header == text_col_name:
                dset_row.append((text_col_name, self.text_field))
            elif header == label_col_name:
                dset_row.append((label_col_name, LabelField(dtype=torch.long)))
            else:
                dset_row.append((header, None))

        dataset = TabularDataset(
            path=self.file_name,
            format="csv",
            fields=dset_row,
            skip_header=True,
            csv_reader_params={"delimiter": delimiter, "quotechar": quotechar}
        )
        return dataset

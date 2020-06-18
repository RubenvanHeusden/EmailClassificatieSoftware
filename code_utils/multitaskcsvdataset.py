"""
This file contains code for loading any dataset that
conforms to the standard csv ('text', 'label') column
format
"""

import torch
import pandas as pd
from torchtext.data import TabularDataset, LabelField


class MultitaskCSVDataset:
    def __init__(self, text_field, path_to_datadir, stratified_sampling=False):
        """

        :param text_field: Object of the torchtext.data.Field type used to load in the text \
        from the dataset

        :param path_to_datadir: path to the directory containing the train, (val), and test files
        """
        self.text_field = text_field
        self.path_to_datadir = path_to_datadir
        self.stratified_sampling = stratified_sampling

    def load(self, text_label_col: str = "text", targets=('label', ), delimiter: str = ",", quotechar: str = '"'):
        field_headers = list(pd.read_csv(self.path_to_datadir, quotechar=quotechar))
        dset_row = []
        for header in field_headers:
            if header == text_label_col:
                dset_row.append((text_label_col, self.text_field))
            elif header in targets:
                dset_row.append((header, LabelField(dtype=torch.long)))
            else:
                dset_row.append((header, None))

        train = TabularDataset(
            path=self.path_to_datadir,
            format="csv",
            fields=dset_row,
            skip_header=True,
            csv_reader_params={"delimiter": delimiter, "quotechar": quotechar}
        )
        return train






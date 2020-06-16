import unittest
import pandas as pd
from torchtext.data import Field
from configurations import ROOT_DIR
from code_utils.csvdataset import CSVDataset


class TestCSVDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = CSVDataset(text_field=Field(), file_name=ROOT_DIR+'/test_data/train.csv')

    def test_data_read(self):
        train_data = pd.read_csv(ROOT_DIR+'/test_data/train.csv')
        output = self.dataset.load()
        self.assertEqual(len(output), len(train_data))
        self.assertTrue(hasattr(output[0], 'text'))
        self.assertTrue(hasattr(output[0], 'label'))

    def test_data_read_altered_col_names(self):
        train_data = pd.read_csv(ROOT_DIR+'/test_data/different_headers.csv')
        self.dataset.file_name = ROOT_DIR+'/test_data/different_headers.csv'
        output = self.dataset.load(text_col_name='tekst', label_col_name='categorie')
        self.assertEqual(len(output), len(train_data))
        self.assertTrue(hasattr(output[0], 'tekst'))
        self.assertTrue(hasattr(output[0], 'categorie'))

    def test_data_read_multiple_cols(self):
        train_data = pd.read_csv(ROOT_DIR+'/test_data/multicol_train.csv')
        self.dataset.file_name = ROOT_DIR+'/test_data/multicol_train.csv'
        output = self.dataset.load()
        self.assertEqual(len(output), len(train_data))
        self.assertTrue(hasattr(output[0], 'text'))
        self.assertTrue(hasattr(output[0], 'label'))
        self.assertFalse(hasattr(output[0], 'col_a'))
        self.assertFalse(hasattr(output[0], 'col_b'))

    def tearDown(self) -> None:
        pass

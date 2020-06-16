import unittest
import pandas as pd
from configurations import ROOT_DIR
from msc.traintestsplitter import TrainTestSplitter


class TestTrainTestSplitter(unittest.TestCase):
    def setUp(self) -> None:
        self.splitter = TrainTestSplitter()

    def test_single_file_split_constant_size(self):
        # Test that the size of the dataframes are correct after splitting
        complete_dataframe = pd.read_csv(ROOT_DIR+'/test_data/data.csv')
        train_dataframe, test_dataframe = self.splitter.stratify_file(ROOT_DIR+'/test_data/data.csv')
        total_len = len(complete_dataframe)
        combined_len = len(train_dataframe) + len(test_dataframe)
        self.assertEqual(total_len, combined_len)

    def test_single_file_split_portion_size(self):
        split_size_train = 0.8
        complete_dataframe = pd.read_csv(ROOT_DIR+'/test_data/data.csv')
        train_dataframe, test_dataframe = self.splitter.stratify_file(ROOT_DIR+'/test_data/data.csv',
                                                                      train_portion=split_size_train)
        total_len = len(complete_dataframe)
        self.assertEqual(len(train_dataframe), int(split_size_train*len(complete_dataframe)))

    def test_incorrect_train_portion(self):
        with self.assertRaises(AssertionError):
            self.splitter.stratify_file(ROOT_DIR+'/test_data/train.csv', train_portion=5.0)
            self.splitter.stratify_file(ROOT_DIR+'/test_data/train.csv', train_portion=0.0)
            self.splitter.stratify_file(ROOT_DIR+'/test_data/train.csv', train_portion=-5.0)

    def tearDown(self) -> None:
        pass

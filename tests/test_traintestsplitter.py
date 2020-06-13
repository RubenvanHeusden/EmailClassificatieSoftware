import unittest
from msc.traintestsplitter import TrainTestSplitter


class TestTrainTestSplitter(unittest.TestCase):
    def setUp(self) -> None:
        splitter = TrainTestSplitter()

    def test_single_file_split(self):
        pass

    def test_reshuffle(self):
        pass

    def test_stratify_func(self):
        pass

    def tearDown(self) -> None:
        pass

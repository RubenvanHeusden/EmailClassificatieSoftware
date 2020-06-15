import unittest
from torchtext.data import Field, Iterator
from code_utils.csvdataset import CSVDataset
from code_utils.dataiterator import DataIterator


class TestDataIterator(unittest.TestCase):
    def setUp(self) -> None:
        text_field = Field()
        self.dataset = CSVDataset(text_field, file_name='../test_data/train.csv').load()
        self.dataset.fields['text'].build_vocab(self.dataset)
        self.dataset.fields['label'].build_vocab(self.dataset)
        self.iterator = DataIterator(Iterator(self.dataset, batch_size=1))

    def test_iterator(self):
        for item in self.iterator:
            self.assertEqual(len(item), 2)

    def tearDown(self) -> None:
        pass

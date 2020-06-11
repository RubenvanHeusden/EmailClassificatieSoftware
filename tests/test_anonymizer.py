import unittest
from msc.anonymizer import Anonymizer


class TestAnonymizer(unittest.TestCase):
    def setUp(self) -> None:
        self.anonymizer_class = Anonymizer()

    def test_anom_string(self):
        test_string = "hallo mijn naam is Bert"
        output = self.anonymizer_class.anonymize_string(test_string)
        self.assertEqual(output, "hallo mijn naam is ")

    def test_correct_file_anom(self):
        self.anonymizer_class.anonymize_file("../test_data/train.csv")

    def test_different_replacement_string(self):
        self.anonymizer_class.set_replacement_string("XXX")
        test_string = "hallo mijn naam is Bert"
        output = self.anonymizer_class.anonymize_string(test_string)
        self.assertEqual(output, "hallo mijn naam is XXX")

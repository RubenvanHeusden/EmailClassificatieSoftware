import unittest
from configurations import ROOT_DIR
from models.pretrainedbert import PretrainedBERT
"""
This file contains several tests to be performed on the PretrainedBERT class.
Because of possible limits on computing power, the train_from_file method is not tested, as this 
could be impossible on a device without a GPU or not enough GPU memory.
"""


class TestPretrainedBert(unittest.TestCase):
    def setUp(self) -> None:
        self.model = PretrainedBERT(path_to_data=ROOT_DIR+"/test_data/train.csv")
        self.model.load_model()

    def test_single_example(self):
        self.model.classify_from_strings("Dit is echt een ontzettend slechte film !")

    def test_list_of_examples(self):
        outputs = self.model.classify_batches(["Dit is echt een ontzettend slechte film !",
                                               "Ik vond de film maar zozo",
                                               "Niet echt de beste film die ik ooit gezien heb, maar ach"])
        self.assertEqual(len(outputs), 3)

    def test_examples_from_file(self):
        outputs = self.model.classify_from_file(file_name=ROOT_DIR+"/test_data/test.csv")
        self.assertEqual(len(outputs), 2)

    def test_loading_error(self):
        model = PretrainedBERT(path_to_data=ROOT_DIR+"/test_data/train.csv")
        with self.assertRaises(AssertionError):
            model.classify_from_strings("Dit gaat niet werken, ik ben vergeten het model eerst te laden")

    def tearDown(self) -> None:
        pass

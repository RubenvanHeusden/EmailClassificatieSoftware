import os
import shutil
import unittest
from configurations import ROOT_DIR
from models.multigatemodel import MultigateModel


class TestMultigateModel(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = MultigateModel(num_outputs_list=[2, 5, 5], target_names_list=["label", "col_a", "col_b"],
                                         max_seq_len=10)

    def test_file_training(self) -> None:
        self.classifier.train_from_file(file_name=ROOT_DIR+"/test_data/multicol_train.csv",
                                        batch_size=1, number_of_epochs=1)
        self.assertTrue(self.classifier.has_trained)

    def test_file_classification(self) -> None:
        self.classifier.train_from_file(file_name=ROOT_DIR+"/test_data/multicol_train.csv", batch_size=1,
                                        number_of_epochs=1)
        predictions = self.classifier.classify_from_file(ROOT_DIR+"/test_data/multicol_test.csv")

    def test_string_or_list_classification(self) -> None:
        self.classifier.train_from_file(file_name=ROOT_DIR+"/test_data/multicol_train.csv", batch_size=1,
                                        number_of_epochs=10)
        outputs = self.classifier.classify_from_strings(["a", "dit is nog een test", "laatse zin"])
        self.assertEqual(len(outputs), 3)

    def test_model_saving(self) -> None:
        self.classifier.save_model(ROOT_DIR+"/test_data/model.pt")

    def test_model_loading(self) -> None:
        self.classifier.save_model(ROOT_DIR+"/test_data/model.pt")
        self.classifier.load_model(ROOT_DIR+"/test_data/model.pt")

    def test_inccorect_model_name_save(self):
        name = "model.p"
        with self.assertRaises(AssertionError):
            self.classifier.save_model(name)

    def test_incorrect_model_name_load(self):
        name = "model.p"
        with self.assertRaises(AssertionError):
            self.classifier.load_model(name)

    def tearDown(self) -> None:
        if os.path.exists(ROOT_DIR+"/test_data/model.pt"):
            os.remove(ROOT_DIR+"/test_data/model.pt")
        if os.path.exists(ROOT_DIR+'/runs/'):
            shutil.rmtree(ROOT_DIR+'/runs/')

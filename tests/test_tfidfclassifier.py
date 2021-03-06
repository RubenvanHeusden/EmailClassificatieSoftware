import os
import sklearn
import unittest
from configurations import ROOT_DIR
from models.tfidf import TFIDFClassifier


class TestTFIDFClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = TFIDFClassifier()

    def test_file_training(self) -> None:
        self.classifier.train_from_file(train_data_path=ROOT_DIR+"/test_data/train.csv")
        sklearn.utils.validation.check_is_fitted(self.classifier.classifier['svc'])

    def test_string_or_list_training(self) -> None:
        list_of_points = [("x_1", "y_1"), ("x_2", "y_2"), ("x_3", "y_3")]
        self.classifier.train_from_strings(list_of_points)
        sklearn.utils.validation.check_is_fitted(self.classifier.classifier['svc'])

    def test_string_training_incorrect_format(self) -> None:
        incorrect_list = [("x_1", "y_1"), ("x_2", "y_2", "z_2")]
        with self.assertRaises(AssertionError):
            self.classifier.train_from_strings(incorrect_list)

    def test_file_classification(self) -> None:
        self.classifier.train_from_file(train_data_path=ROOT_DIR+"/test_data/train.csv")
        predictions = self.classifier.classify_from_file(ROOT_DIR+"/test_data/test.csv")
        self.assertEqual(len(predictions), 2)

    def test_string_or_list_classification(self) -> None:
        list_of_points = [("x_1", "y_1"), ("x_2", "y_2"), ("x_3", "y_3")]
        sample_point = ["x_4"]
        self.classifier.train_from_strings(list_of_points)
        self.classifier.classify_from_strings(sample_point)

    def test_scoring(self) -> None:
        self.classifier.train_from_file(train_data_path=ROOT_DIR+"/test_data/train.csv")
        self.classifier.score(ROOT_DIR+"/test_data/test.csv")

    def test_model_saving(self) -> None:
        self.classifier.save_model(ROOT_DIR+"/test_data/model.joblib")

    def test_model_loading(self) -> None:
        self.classifier.save_model(ROOT_DIR+"/test_data/model.joblib")
        self.classifier.load_model(ROOT_DIR+"/test_data/model.joblib")

    def tearDown(self) -> None:
        if os.path.exists(ROOT_DIR+"/test_data/model.joblib"):
            os.remove(ROOT_DIR+"/test_data/model.joblib")

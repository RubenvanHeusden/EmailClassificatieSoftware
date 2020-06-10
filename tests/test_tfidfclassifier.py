import os
import sklearn
import unittest
from models.tfidf import TFIDFClassifier


class TestTFIDFClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = TFIDFClassifier()

    def test_file_training(self) -> None:
        self.classifier.train_from_file(train_data_path="../test_data/train.csv")
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
        self.classifier.train_from_file(train_data_path="../test_data/train.csv")
        predictions = self.classifier.classify_from_file("../test_data/test.csv")
        self.assertEqual(len(predictions), 2)

    def test_string_or_list_classification(self) -> None:
        list_of_points = [("x_1", "y_1"), ("x_2", "y_2"), ("x_3", "y_3")]
        sample_point = ["x_4"]
        self.classifier.train_from_strings(list_of_points)
        self.classifier.classify_from_strings(sample_point)

    def test_scoring(self) -> None:
        self.classifier.train_from_file(train_data_path="../test_data/train.csv")
        self.classifier.score("../test_data/test.csv")

    def test_model_saving(self) -> None:
        self.classifier.save_model("../test_data/model.joblib")

    def test_model_loading(self) -> None:
        self.classifier.save_model("../test_data/model.joblib")
        self.classifier.load_model("../test_data/model.joblib")

    def test_inccorect_model_name_save(self):
        name = "model.jobli"
        with self.assertRaises(AssertionError):
            self.classifier.save_model(name)

    def test_incorrect_model_name_load(self):
        name = "model.jobli"
        with self.assertRaises(AssertionError):
            self.classifier.load_model(name)

    def tearDown(self) -> None:
        if os.path.exists("../test_data/model.joblib"):
            os.remove("../test_data/model.joblib")

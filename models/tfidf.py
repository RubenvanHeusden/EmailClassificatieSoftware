# This file contains the implementation of a TF-IDF model followed by a Support Vector Machine
# Classifier. The Support Vector Model performed the best in comparison to other models combined with TF-IDF
# As can be read in the technical report. the model uses the parameters found to be best for the email classification
# dataset.

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from joblib import dump, load
from typing import List, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class TFIDFClassifier:
    """
    class implementing a TF-IDF classifier combined with an SVM model

    Attributes
    ----------
    classifier : sklearn.pipeline.Pipeline
        Pipeline object used for the preprocessing and classification of the data. The specific
        models and the reason they were chosen can be found in the technical report

    string_encoding : str
        string representing the type of encoding to be used when reading the string, the default is 'utf-8'


    _name : string
        string representation of the name of the model to be used for string representation

    """
    def __init__(self, do_lowercase: bool = True, string_encoding: str = 'utf-8',
                 custom_tokenizer=None, custom_preprocessor=None, n_grams=(1, 1), verbose_training: bool = False,
                 use_confidence: bool = False):
        """
        :param do_lowercase: boolean specifying whether text should be automatically converted to lowercase
        :param string_encoding: specifying the encode to use when reading in text. default = 'utf-8'
        :param custom_tokenizer: if not None, a callable that takes string input and returns a list of tokens
        :param custom_preprocessor: of not None, a callable that takes string input and return a string \
        processed in some way. (see the examples folder for example use-cases)
        :param n_grams: a tuple specifying the ngrams range used. default is (1, 1)
        :param use_confidence: boolean specifying whether to calculate confidence scores for outputs, this \
        requires the classes in the training file to all have at least 5 examples and the confidence scores \
        in the classify methods will still have to be set manually
        """
        self._name = "TFIDF"
        if not use_confidence:
            cls = LinearSVC(verbose=verbose_training)
        else:
            cls = CalibratedClassifierCV(LinearSVC(verbose=verbose_training))

        self.classifier = Pipeline([('tfidf', TfidfVectorizer(lowercase=do_lowercase,
                                                              encoding=string_encoding,
                                                              preprocessor=custom_preprocessor,
                                                              tokenizer=custom_tokenizer,
                                                              ngram_range=n_grams)),
                                    ('svc', cls)])
        self.string_encoding = string_encoding

    def save_model(self, file_name: str) -> None:
        """

        saves the (trained) classifier from sklearn using the joblib module

        :param file_name: string specifying the name of the file to where \
        the model is saved. the standard format is 'joblib' and should be specified.

        """
        assert file_name.split('.')[-1] == "joblib"
        dump(self.classifier, file_name)
        return None

    def load_model(self, file_name: str) -> None:
        """

        load a classifier saved by the save_model method

        :param file_name: string specifying the name of the file from which the model \
        is loaded. the standard format is 'joblib' and should be specified.

        """
        assert file_name.split('.')[-1] == "joblib"
        self.classifier = load(file_name)
        return None

    def train_from_file(self, train_data_path, text_col_name: str = "text", label_col_name: str = "label",
                        delimiter=",", quotechar='"') -> None:
        """

        trains a model with data loaded from a csv file where the columns specifying the text and labels
        are arguments to the function


        :param train_data_path: string specifying the file that contains the data points that the classifier is \
         trained on
        :param text_col_name: string specifying the name of the column containing the mails in the csv file
        :param label_col_name: string specifying the name of the column containing the labels of the mails in the \
         csv file
        :param delimiter: string specifying the delimiter of the csv file, defaults to ","
        :param quotechar: specifying the character as quotechar in the csv reader. defaults to "

        """
        training_data = pd.read_csv(train_data_path, encoding=self.string_encoding, quotechar=quotechar,
                                    sep=delimiter)
        training_data = training_data.dropna()
        training_data_x = training_data[text_col_name].tolist()
        training_data_y = training_data[label_col_name].tolist()
        assert len(training_data_x) == len(training_data_y)
        self.classifier.fit(training_data_x, training_data_y)
        return None

    def train_from_strings(self, train_data: List[Tuple]) -> None:
        """

        trains a model from an input list 'train_data' of tuples where each tuple specifies a data point

        :param train_data: list of tuples of the form [(x_1, y_1), (x_2, y_2), ..., (x_n, y_n)] \
        specifying the text label pairs used for training.

        """
        assert all([True if len(point) == 2 else False for point in train_data])

        training_x = [data_point[0] for data_point in train_data]
        training_y = [data_point[1] for data_point in train_data]
        self.classifier.fit(training_x, training_y)
        return None

    def classify_from_file(self, classification_data_path, text_col_name: str = "text", delimiter=",",
                           quotechar='"', confidence_threshold: float = 0.0) -> List[Any]:
        """

        classifies examples from a csv file with the trained classifier, where the column with the text
        to be classified is selected with the text_col_name parameter

        :param confidence_threshold: float specifying the desired confidence when outputting a prediction. \
        when not set to zero, will only return predictions when score > confidence_threshold, None otherwise. \
        please note that using this threshold will slow down the classification speed of the algorithm \
        the confidence_threshold is a probability and should therefore lie between 0 and 1.
        :param classification_data_path: string specifying the file that contains the \
        texts to be classified by the classifier
        :param text_col_name: string specifying the name of the column containing the mails \
        in the csv file
        :param delimiter: string specifying the delimiter of the csv file, defaults to ","
        :param quotechar: specifying the character as quotechar in the csv reader. defaults to "
        :return: A list of predictions with the same length as the number of data points in the file, \
        where each entry in the list is a prediction of the model

        """

        assert 0.0 <= confidence_threshold < 1.0

        classification_data = pd.read_csv(classification_data_path, sep=delimiter, quotechar=quotechar)
        classification_data = classification_data.dropna()
        classification_x = classification_data[text_col_name].tolist()

        predictions = self.classifier.predict(classification_x)

        if confidence_threshold > 0.0:
            scores = np.max(self.classifier.predict_proba(classification_x), axis=1)
            predictions = [item if score > confidence_threshold else None for item,
                                                                              score in zip(predictions, scores)]

        return predictions

    def classify_from_strings(self, classification_data, confidence_threshold: float = 0.0) -> List[Any]:
        """

        classifies a single example or a list of examples with the trained classifier

        :param confidence_threshold: float specifying the desired confidence when outputting a prediction. \
        when not set to zero, will only return predictions when score > confidence_threshold, None otherwise. \
        please note that using this threshold will slow down the classification speed of the algorithm \
        the confidence_threshold is a probability and should therefore lie between 0 and 1.
        :param classification_data: string or list containing the example(s) for classification. \
        the acceptance of a string to signify a single example is done in favor of manually putting a single \
        example into a list.
        :return: A list of strings with the same length as the input list with the predictions of the model
        """

        assert 0.0 <= confidence_threshold < 1.0

        if isinstance(classification_data, str):
            classification_data = [classification_data]

        predictions = self.classifier.predict(classification_data)

        if confidence_threshold > 0.0:
            scores = np.max(self.classifier.predict_proba(classification_data), axis=1)
            predictions = [item if score > confidence_threshold else None for item,
                                                                      score in zip(predictions, scores)]

        return predictions

    def score(self, test_data_path, text_col_name: str = "text", label_col_name: str = "label",
              delimiter=",", quotechar='"', verbose=1, class_averaging: str = "weighted") -> None:
        """

        score(test_data_path, text_col_name, label_col_name, delimiter, verbose, class_averaging)

        :param test_data_path: string specifying the file that contains the data points that have to be \
        scored
        :param text_col_name: string specifying the name of the column containing the mails \
        in the csv file
        :param label_col_name: string specifying the name of the column containing the labels of the mails \
        in the csv file
        :param delimiter: string specifying the delimiter of the csv file, defaults to ","
        :param quotechar: specifying the character as quotechar in the csv reader. defaults to "
        :param verbose: integer specifying the verbosity of the returned score, below are the possible \
        options: \
            1: return only accuracy \
            2: return overall accuracy, precision, recall and f1 \
            3: return detailed sklearn classification report \
        Because the dataset is multiclass and there is possible class imbalance, weighted versions \
        of all classification scores are used
        :param class_averaging: specifies the kind of averaging to use for the scoring in a multiclass \
        classification setting. see Sklearn documentation for more details on the options.
        :return: scores indicating the performance of the model, depends on level of verbosity set.

        """

        testing_data = pd.read_csv(test_data_path, sep=delimiter, quotechar=quotechar)
        testing_data = testing_data.dropna()

        testing_data_x = testing_data[text_col_name].tolist()
        testing_data_y = testing_data[label_col_name].tolist()

        assert len(testing_data_x) == len(testing_data_y)

        predictions = self.classify_from_file(test_data_path, text_col_name=text_col_name,
                                              delimiter=delimiter, quotechar=quotechar)
        if verbose == 1:
            print("Accuracy score of the model is %.3f" % accuracy_score(testing_data_y, predictions))
        elif verbose == 2:
            print("Performance of the model: \n"
                  "Accuracy: %.3f \n"
                  "Precision: %.3f \n"
                  "Recall: %.3f \n"
                  "F1: %.3f \n" % (
                    accuracy_score(testing_data_y, predictions),
                    precision_score(testing_data_y, predictions, average=class_averaging),
                    recall_score(testing_data_y, predictions, average=class_averaging),
                    f1_score(testing_data_y, predictions, average=class_averaging)))
        elif verbose == 3:
            print(classification_report(testing_data_y, predictions, digits=3))
        return None

    def __repr__(self):
        return self._name

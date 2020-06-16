"""
This file contains a few small examples uses of the anonymization class.
It demonstrates its basic use and the specific usage of the experimental parallel anonymization
function.
"""

from configurations import ROOT_DIR
from msc.anonymizer import Anonymizer
from models.tfidf import TFIDFClassifier


def main():
    # First we create an instance of the Anonymizer class

    anonymizer_class = Anonymizer()

    # Now we can use the algorithm on a single sentence

    example_sentence = "Hallo mijn naam is Bert and ik woon aan de Kalverstraat"
    print(anonymizer_class.anonymize_string(example_sentence))

    # As can be seen from the example, the standard behaviour of the anonymizer is to replace
    # the named entity found in the text with "", effectively removed it. Although this behaviour
    # is ok for most use cases, this can be altered if needed.

    anonymizer_class.set_replacement_string("XXX")

    # Now we can try the anonymization again and see the difference
    print(anonymizer_class.anonymize_string(example_sentence))

    # The anonymize_from_file function can be used when a complete csv file has to be anonymized
    data_file = ROOT_DIR+"/test_data/train.csv"
    print(anonymizer_class.anonymize_file(data_file))

    # Although the Anonymizer is fully functional as a standalone class, it can also be used in combination with
    # Most of the models in this module. As an example, we can use the 'anonymize_string' method of the anonymizer
    # in the preprocessing step of the TF-IDF classifier to filter the emails before they get fed into the
    # classifier

    classifier = TFIDFClassifier(custom_preprocessor=anonymizer_class.anonymize_string)
    path_to_data = ROOT_DIR+"/test_data/train.csv"
    classifier.train_from_file(path_to_data, text_col_name='text', label_col_name="label")
    classifier.score(ROOT_DIR+"/test_data/test.csv", verbose=1)

    # It is however recommended that for large files, the anonymization is
    # done once and the results stored in a separate file because the process is quite time consuming for large
    # files


if __name__ == "__main__":
    main()

"""
This file contains example code for several features of the TF-IDF + SVM classification model
It shows how to setup and train the classifier from scratch either from datafiles are from a list of datapoints.
It also shows how the model can be saved once trained and reloaded for later use.
"""
from models.tfidf import TFIDFClassifier

# Setting up the model

# Make an instance of the TFIDF Classifier (the default arguments will work for most cases)
# For demonstration purposes we set the verbose mode of the model to true to show progress during training
classifier = TFIDFClassifier(verbose_training=True)

# A model can either be trained from a csv file or from a list of data points (see the documentation
# for the TF-IDF model for more details)
path_to_data = "../test_data/train.csv"


# It is important here that the file has headers that specify which data should be used.
# in our example the data file has a column 'text' for the email texts and a column 'label'
# for the labels associated with the text so we give those column names as arguments to the
# train_from_file method of the TF-IDF classifier.

# Training the model

classifier.train_from_file(path_to_data, text_col_name='text', label_col_name="label")

# Now that the classifier is trained on the dataset, we can either classify a file with data in the
# same format as the training csv file, or classify a separate piece of text. In the case of a piece
# of text, a single sentence mail can be fed in as a string, or a list of strings can be fed in.


# Classifying examples
example_sentence = "Dit is echt de beste film die ik ooit gezien heb"
print(classifier.classify_from_strings(example_sentence))

# Evaluating the model

# Although the classify_from_strings and classify_from_file methods are most likely the methods
# that will are needed, there is also the possibility to evaluate the quantative performance of the model.
# As this is likely something that will be done using a test file, this method is only implemented for
# reading from a file.

classifier.score(test_data_path="../test_data/test.csv", verbose=3)

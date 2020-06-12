# Quick testing of some of the major features of the module
from models.tfidf import TFIDFClassifier

# Train the model

classifier = TFIDFClassifier()
classifier.train_from_file("test_data/train.csv")


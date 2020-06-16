Getting Started
***************

The EmailClassificatieSoftware package contains several implementations of current state-of-the-art
algorithms for text classification, adapted for use with the Dutch language. The main usage of this 
is for either training one of the algorithms in the package from scratch for a new dataset or task,
or as part of another tool where pretrained models can be loaded and easily used for classification.

Training a simple classifier from scratch
=========================================

To show the main features of this package we will start by training a simple TF-IDF classifier from scratch.
All the models in this package expect to receive data either from csv files or directly from a list of strings.
If you already have a dataset splitted into a train and test set you can use these directly by giving 
the path of files to the model when creating it. However, if you do not already have separate train and test files
but do have a csv file containing the text you want to classify and the appropriate labels for these pieces of text,
you can use the TrainTestSplitter class from msc folder to automatically create these for you.(See the documentation
of the TrainTestSplitter for more information on this).

First we will import the TFIDF classifier into our script and create and instance of it that we will use for classification
``classifier = TFIDFClassifier(verbose_training=False)``

here the `` verbose_training=False`` indicates that we don't want to see intermediate statistics of training, this can be 
quite a lot of text, so we will turn it off for this example.

Now that we have created a classifier, it is time to train the model on a dataset!

To do this we can simple specify the path to to our file with train data. 

``classifier.train_from_file('test_data/train.csv', text_col_name='text', label_col_name="label")``

Here the 'text_col_name' and 'label_col_name' indication the name of the header of the column in the csv file
containing the text and the labels respectively.

Now that we have trained our classifier we can start classifying some inputs!

To do this we can simply call ``classifier.classify_from_strings("Een stukje tekst voor classificatie")``.


Loading and saving models
=========================


And that's it! we have trained simple TFIDF classifier, classified an example and saved the model for later use.
The other models in this module also follow the same workflow, so they can be trained in much the same way as 
the TF-IDF classifier.

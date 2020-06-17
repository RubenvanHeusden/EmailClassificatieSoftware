Utilities
*********


Text Anonymization
==================

DISCLAIMER: The anonymization algorithm using spaCy in this research is 
not tested properly on a large dataset for its ability to anonymize where
detailed statistics about its behaviour could be recorded. Although usage in the 
research showed it appeared to be very well in anonymization, this is not a guarantee
and the text in the research was checked and corrected manually afterwards.

This package comes with an anonymization algorithm implemented in the :class:`.Anonymizer`
class based upon the Named Entity Recognition (NER) detection algorithm for Dutch by spaCy.

To use the anonymizer we create a class and set some parameters

.. code-block::

	anonymizer = Anonymizer`(replacement_string="X")

Here the ``replacement_string="X"`` indicates that we want to replace any occurance of a named entity
in our piece of text with the character "X".

Now the we have created the anonymizer we can directly anonymize a string via the :meth:`.Anonymizer.anonymize_string`
method.

.. code-block::

	anonymizer.anonymize_string("Hallo mijn naam is Bert and ik woon aan de Kalverstraat")

Although this method works fine, you probably already noticed that it takes quite a while to anonymize even a single sentence.
For this reason it is generally recommended to use the :meth:`.Anonymizer.anonymize_file` method to anonymize all the data you
want to anonymize before training and saving this to a file, instead of using this method as a preprocessor in the training scripts
of the models.


Train and Test File Splitting
=============================
If you want to train a model but you do not yet have separate train and test files, or your label distribution
in your train and test files is very different, you can use the :class:`.TrainTestSplitter` class to create 
separate train and test files or reshuffle exising train and test files.

First we will instantiate a traintestsplitter

.. code-block::

    splitter = TrainTestSplitter()


Now the we have created our splitter we can use it to split an existing file into two dataframes which can then
be written to separate files


.. code-block::
	
	train_dataframe, test_dataframe = splitter.stratify_file(file_name="/test_data/data.csv")

Now we can also reshuffle existing train and test files by using the :method:'.TrainTestSplitter.reshuffle' method
    
.. code-block::

	train_set, test_set = splitter.reshuffle('/test_data/train.csv', '/test_data/test.csv')


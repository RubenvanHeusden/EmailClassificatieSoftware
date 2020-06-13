"""
This file contains several examples on how to use the traintestsplitter to split a file containing data
into separate train and test files and on how to use the 'reshuffle' method of the splitter to reshuffle
existing train and test files.
"""

from msc.traintestsplitter import TrainTestSplitter

# First we instantiate a class of the TrainTestSplitter to use for our splitting
splitter = TrainTestSplitter()

# First we will split the 'data.csv' file in the 'test_data' directory into two separate files and
# check that their distribution is (almost) similar.

data_fil_path = "../test_data/data.csv"

# Now we will split the file into a train and test set, wher the 'stratify_file' method
# returns two dataframes
train_dataframe, test_dataframe = splitter.stratify_file(file_path=data_fil_path)

# We can print the distribution of the labels of the dataset and see that they are close.
# (the stratify function uses a 70/30 split for train and test by default)
print(train_dataframe['label'].value_counts())
print(test_dataframe['label'].value_counts())
print()
# Now we can also reshuffle existing train and test files by using the 'reshuffle' method
print("Reshuffling the train and test files")
train_dataframe, test_dataframe = splitter.reshuffle('../test_data/train.csv', '../test_data/test.csv')
print(train_dataframe['label'].value_counts())
print(test_dataframe['label'].value_counts())
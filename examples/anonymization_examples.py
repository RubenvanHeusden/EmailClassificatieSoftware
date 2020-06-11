"""
This file contains a few small examples uses of the anonymization class.
It demonstrates its basic use and the specific usage of the experimental parallel anonymization
function.
"""

from msc.anonymizer import Anonymizer

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
data_file = "../test_data/train.csv"
print(anonymizer_class.anonymize_file(data_file))

# Although the above function is adequate for most use cases, it can be quite slow for large files.
# for these cases the anonymize_parallel function can be used to speed up this process.
# PLEASE NOTE THAT THIS FUNCTION IS EXPERIMENTAL AND NOT THOROUGHLY TESTED, UNEXPECTED ERRORS MAY OCCUR

# For this function to work on Windows it is important that the anonymization happens in a separate Python file
# and it should be called in the following way:

# if __name__ == "__main__":
#     anonymizer_class._anonymize_parallel(file_name=data_file)

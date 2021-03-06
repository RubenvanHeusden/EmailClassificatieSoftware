"""
This file contains a class for the rudimentary anonymization of sensitive data.
The class uses a Named Entity Recognition algorithm from spaCy to achieve this.
The model achieves a very high recall rate meaning that a large amount of the
named entities get removed. Although this is desirable from the security perspective,
this does mean that the model will have a significant false positive ratio where words
are mistakenly seen as a Named Entity and wrongfully removed. Testing showed that this
did not have a significant impact on the results of the models used in this research.

------------------------------------ DISCLAIMER ------------------------------------
    Although the anonymization algorithm implemented in this file is used because
    of its ability to obtain a high recall, it is by no means perfect and it can
    therefore not be guaranteed that all privacy sensitive information is removed.
    It is STRONGLY RECOMMENDED that this tool is used as a rough first step in
    the anonymization process, after which the output is checked and where needed
    corrected by a human.
------------------------------------------------------------------------------------
"""

import spacy
import pandas as pd


class Anonymizer:
    """
    This class implements a automatic anonymization algorithm for Dutch text using the
    spaCy Named Entity Recognition algorithm

    Attributes
    ----------
    nlp_model: spacy_model
        the subset of spaCy models for the language passed in as parameters, the Dutch version
        is used in this class

    """
    def __init__(self, replacement_string: str = ""):
        """
        :param replacement_string: string that is used to replace any Named Entity found
        """
        self.nlp_model = spacy.load('nl_core_news_sm')
        self.replacement_string = replacement_string

    def anonymize_string(self, string: str) -> str:
        """

        method that performs anonymization for a single string (this method is also used by the
        anonymize file method)

        :param string: string to be anonymized
        :return: string anonymized by the spaCy Named Entity Recognition algorithm
        """
        doc = self.nlp_model(string)
        for ent in doc.ents:
            string = string.replace(ent.text, self.replacement_string)
        return string

    def anonymize_file(self, file_name, delimiter: str = ",", quotechar: str = '"',
                       text_col_name: str = "text") -> pd.DataFrame:

        """

        convenience method to anonymize all the text entries in a file.


        :param file_name: string specifying the name of the csv file which contains the data to be anonymized.
        :param delimiter: the delimiter used for the reading of the csv file, default is ','
        :param quotechar: the quotation character used by the csv reader, default is '"'
        :param text_col_name: string signifying the name of the column in the csv file containing the text \
        that is te be anonymized.
        :return csv_file: returns the pandas DataFrame as specified by the filename parameter with the text \
        column anonymized.

        """
        csv_file = pd.read_csv(file_name, sep=delimiter, quotechar=quotechar)
        unfiltered_text = csv_file[text_col_name]
        filtered_text = unfiltered_text.apply(self.anonymize_string)
        csv_file[text_col_name] = filtered_text
        return csv_file

    def get_replacement_string(self) -> str:
        """

        returns the string currently set as the replacement string for named entities

        :return: return the current replacement string for Named Entities
        """
        return self.replacement_string

    def set_replacement_string(self, replacement_string: str) -> None:
        """

        set the argument replacement_string as the string to use when removing named entities

        :param replacement_string: string that is to be set as the replacement string \
        in the algorithm
        """
        self.replacement_string = replacement_string
        return None

"""
This file implements a BERT model from the huggingface-transformers Library which is fine-tuned on the
specific dataset. As the training procedure is relatively involved, two versions of this model are included in this
module. The one implemented in this file is an instantiation of the trained Bert model where the trained model is
simply loaded and input data is simply converted to the right format and fed through the model.
"""

import torch
import pandas as pd
from tqdm import tqdm
from typing import List
from transformers import BertForSequenceClassification, BertTokenizer


class PretrainedBert:
    """
    This class implements an interface to a pretrained model on Dutch data, handling the tokenization of the text
    and te conversion of the output of the model to the appropriate classes. This class can be used interact
    with a model trained using the Bert model in this module.

    Attributes
    ----------
    device: torch.device
        if use_gpu is set to true and a gpu is available on the system the model and inputs will be run on
        the gpu. This is significantly faster than running the model on the gpu.

    bert_model
        An instance of the BertForSequenceClassification model that is trained on the Dutch Language
        and used to classify the examples

    bert_tokenizer
        tokenizer for the BertModel this is trained on the Dutch language and used to convert sequences of strings
        to sequences of indices into the sentencepiece embeddings model of Bert.

    _class_labels
        labels that are used in this specific research, the labels should match the labels present in the
        dataset
    """
    def __init__(self, path_to_model: str, use_gpu: bool = False, path_to_data: str = "../data/train.csv"):

        self.device = torch.device("cuda" if (torch.cuda.is_available() and use_gpu) else "cpu")

        self.bert_model = BertForSequenceClassification.from_pretrained(path_to_model).to(self.device)
        self.bert_tokenizer = BertTokenizer.from_pretrained(path_to_model)

        self._class_labels = pd.read_csv(path_to_data, sep=",", quotechar='"')['label'].unique()

    def classify_from_strings(self, text_string: str) -> List:
        """
        :param text_string: string that contains the text to be classified
        :return: list containing the string representation of the most likely class according
        to the Bert model.
        """
        input_ids = torch.tensor([self.bert_tokenizer.encode(text_string, add_special_tokens=True)]).to(self.device)
        with torch.no_grad():
            model_output = self.bert_model(input_ids)
            return self._class_labels[model_output[0].argmax()]

    def classify_from_file(self, file_name: str, delimiter: str = ",", quotechar: str = '"',
                           text_col_name: str = 'text', batch_size: int = 8) -> List:
        """
        Convenience function to classify the contents of a csv file. This function will call
        'classify_batches' internally to do the classification

        :param file_name: string indicating the csv file containing the text to be classified
        :param delimiter: delimiter used in the csv file, to read the csv file
        :param quotechar: quotechar used in the csv file, to read  the csv file
        :param text_col_name: string signifying the name of the column in the csv file containing the
        text to be classified
        :param batch_size: amount of examples to feed to the network simultaneously, this
        can be lowered if memory issues are encountered
        :return: list of predictions for the text found in the csv file
        """
        text_from_file = pd.read_csv(file_name, sep=delimiter, quotechar=quotechar)[text_col_name]
        text_from_file = text_from_file.dropna().tolist()
        batches = [text_from_file[i:i + batch_size] for i in range(0, len(text_from_file), batch_size)]
        predictions = []
        for batch in tqdm(batches):
            predictions.extend(self.classify_batches(batch))
        return predictions

    def classify_batches(self, list_of_examples: List) -> List[str]:
        """
        This method implements a way of classifying a list of examples by feeding batches into the network.
        This is significantly more efficient for classifying a list of examples than calling the
        "classify_from_strings" method on each item in the list
        :param list_of_examples: list of strings containing the texts to be classified by the Bert model
        :return: list of predictions with the same length as the number of items in 'list_of_examples'
        """
        encoded_batch = self.bert_tokenizer.batch_encode_plus(list_of_examples, add_special_tokens=True,
                                                               return_tensors='pt', pad_to_max_length=True,
                                                              max_length=100)
        with torch.no_grad():
            model_output = self.bert_model(encoded_batch['input_ids'].to(self.device),
                                           attention_mask=encoded_batch['attention_mask'].to(self.device))

            predicted_classes = model_output[0].argmax(1)
            return [self._class_labels[i] for i in predicted_classes]
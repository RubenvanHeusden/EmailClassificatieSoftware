"""
This file contains the implementation of the CNN Classifier class. This class is essentially a wrapper
class around the CNN model, handling the loading of the datafiles and the setting up of the training
procedure.
"""
import torch
import torchtext
import pandas as pd
import torch.nn as nn
from models.cnn import CNN
import torch.optim as optim
from typing import List, Union
from configurations import ROOT_DIR
from torch.optim.lr_scheduler import StepLR
from code_utils.csvdataset import CSVDataset
from code_utils.dataloader import CustomDataLoader
from torchtext.data import Example, Iterator, Field
from code_utils.utils import embeddings_available, download_word_embeddings_nl, single_task_class_weighting,\
    generic_training, generic_evaluation


class CNNClassifier:
    """
    This class implements a CNN Classifier based on the 'CNN for text classification' from Yoon Kim \
    It deals with the various aspects of the training, such as converting the data into the appropriate \
    format and logging the training process via TensorBoard

    Attributes
    ----------
        device: torch.device
            torch.device indicating on which device the model and the inputs should be, either on the GPU or the
            CPU. The default behaviour is to put the model and the inputs on the GPU when available.

        model: nn.Module
            The main model used for classification, in this case the CNN model

        num_outputs: int
            Integer specifying the number of outputs of the model. This should be set to the number of unique classes
            in the dataset. (the 'get_num_labels_from_file' method can be used to retrieve this from the csv file
            when this is not known)

        has_trained: bool
            Boolean specifying whether the model has already been trained. This is used to ensure that the evaluaton
            or scoring is not accidentally run on an untrained model.

        _TEXT: torchtext.data.Field
            torchtext.data.Field instance specifying several parameters of the reading of the data such as
            whether or not to convert all text to lowercase and the type and language of the tokenizer used.

        _words: list
            list with all the words present in the Dutch embedding file

        _embed_dict: dict
            dictionary mapping words in the embeddings file to indices into the embedding matrix

        _embeddings: torch.Tensor
            torch.Tensor of size [num_words, embedding_dim] containing the word embeddings

        _criterion nn.optim.Criterion
            criterion used for the training and evaluation of the model. This is saved in the train methods
            for later use in the evaluation methods

        _embed_dim: int
            Integer specifying the dimension of the embeddings used in the embedding file

        _label_names: list
            list containing the names of the unique labels in the dataset, this is used for converting the
            integer representation used in training back to the original labels for easier interpretation

    """

    def __init__(self, num_outputs, num_filters: int = 100,
                 filter_list: tuple = (3, 4, 5), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 word_embedding_path: str = ROOT_DIR+'/resources/word_embeddings/combined-320.tar/320/',
                 max_seq_len=None, dropout: float = 0.5):
        """
        :param num_filters: integar specifying the number of filters used for each size of filter
        :param filter_list: tuple containing the sizes of the filters used in the model
        :param num_outputs: integer specifying the number of outputs of the model, when unknown in advance, \
        this can be retrieved by using the 'get_num_labels_from_file' method
        :param device: torch.device specifying the device on which the inputs and the model should be put. \
        By default the model will be put on the GPU if one is available
        :param word_embedding_path: string specifying the path of the word embedding text and pt files
        :param max_seq_len: the maximum length to which sentences are clipped, this can be used when some \
        sentence are very long, which can cause memory issues when using larger batch sizes.
        :param dropout: float specifying the amount of dropout that is applied to the penultimate layer of the model
        """
        # Load in the vectors when they are not already present in the package
        if not embeddings_available():
            download_word_embeddings_nl()
            print("--- Constructing the Pytorch embedding matrix file ---")
            torchtext.vocab.Vectors('combined-320.txt', cache=word_embedding_path)

        vocab_data = torch.load(word_embedding_path+"combined-320.txt.pt")

        self.device = device

        self._words, self._embed_dict, self._embeddings, self._embed_dim = vocab_data

        self.model = CNN(input_channels=1, output_dim=num_outputs, filter_list=filter_list,
                         embed_matrix=torch.zeros(size=(1, 1)), num_filters=num_filters, dropbout_probs=dropout)

        self._TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="nl_core_news_sm", include_lengths=False,
                           batch_first=True, fix_length=max_seq_len)

        self.num_outputs = num_outputs

        self._criterion = None

        self._label_names = None

        self.has_trained = False

    def train_from_file(self, file_name: str, batch_size: int, num_epochs: int, delimiter: str = ",",
                        quotechar: str = '"', text_col_name: str = 'text', label_col_name='label', learning_rate=1.0,
                        logging_dir: str = ROOT_DIR+'/runs/') -> None:
        """

        The main method of this class, implementing a training procedure for the model and handling
        the proper loading of the dataset


        :param file_name: string specifying the location and name of the file that contains the training dat
        :param batch_size: integer specifying the batch size, this will affect the size of the batches fed into \
        the model this can be set lower if memory issues occur
        :param num_epochs: integer specifying the number of epochs for which the model is trained. The right amount of \
        epochs can differ for different datasets and it is recommended to inspect the produced TensorBoard logs \
        to see if the model has converged
        :param delimiter: string specifying the delimiter used in the training csv file
        :param quotechar: string specifying the quotechar used in the training csv file
        :param text_col_name: string specifying the name of the column containing the mails in the csv file \
        :param label_col_name: string specifying the name of the column containing the labels of the mails in the \
        csv file
        :param learning_rate: float specifying the learning rate of the model, this can affect the speed of \
        convergence of the model
        :param logging_dir: directory to which the Tensorboard logging files are saved

        """
        print("--- Starting with reading in the dataset ---")
        dataset_loader = CSVDataset(text_field=self._TEXT, file_name=file_name)
        dataset = dataset_loader.load(delimiter=delimiter, quotechar=quotechar, text_col_name=text_col_name,
                                      label_col_name=label_col_name)
        print("--- Finished with reading in the dataset ---")

        dloader = CustomDataLoader(dataset)
        data_iterator = dloader.construct_iterators(batch_size=batch_size, text_col_name=text_col_name,
                                                    label_col_name=label_col_name)

        self._TEXT.vocab.set_vectors(self._embed_dict, self._embeddings, self._embed_dim)

        self.model.set_new_embedding_matrix(self._TEXT.vocab.vectors)
        self._label_names = dataset.fields[label_col_name].vocab.itos

        weights = single_task_class_weighting(data_iterator)
        criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        self._criterion = criterion

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

        generic_training(self.model, criterion, optimizer, scheduler, data_iterator, device=self.device,
                         tensorboard_dir=logging_dir, n_epochs=num_epochs, clip_val=0.0)

        self.has_trained = True
        return None

    def classify_from_file(self, file_name, delimiter: str = ",", quotechar: str = '"', text_col_name: str = "text",
                           batch_size: int = 64) -> list:
        """
        This method reads in a file, parses it into the correct format and classifies the contents
        of the file. Throws an error when the model is not trained.

        :param file_name: string specifying the location and name of the file that contains the training dat
        :param delimiter: string specifying the delimiter used in the training csv file
        :param quotechar: string specifying the quotechar used in the training csv file
        :param text_col_name: string specifying the name of the column containing the mails in the csv file
        :param batch_size: integer specifying the batch size, this will affect the size of the batches fed into \
        the model this can be set lower if memory issues occur
        :return: returns a list of results, where the result indices from the model have been converted back to \
        the original class names from the file
        """
        assert self.has_trained

        strings = pd.read_csv(file_name, sep=delimiter, quotechar=quotechar)[text_col_name].tolist()

        if isinstance(strings, str):
            strings = [strings]
        if isinstance(strings, list):
            strings = [[string] for string in strings]

        fields = [('text', self._TEXT)]

        list_of_examples = [Example.fromlist(string, fields) for string in strings]
        dataset = torchtext.data.Dataset(list_of_examples, fields)

        data = Iterator(dataset, batch_size=batch_size, device=torch.device("cpu"), sort=False, sort_within_batch=False,
                        repeat=False, shuffle=False)

        predictions = []

        for item in data:
            x = item.text
            self.model.to(self.device)
            self.model = self.model.eval()
            outputs = self.model(x.to(self.device))
            predictions.extend(outputs.detach().cpu().argmax(1).tolist())
        results = [self._label_names[i] for i in predictions]
        return results

    def classify_from_strings(self, strings: Union[List[str], str]) -> list:
        """

        method that can be used for classifying one or multiple examples with a trained classifier

        :param strings: a single string or a list of strings representing the pieces of text that should be classified
        :return: list containing the predictions of the models for the inputted pieces of text
        """
        assert self.has_trained
        if isinstance(strings, str):
            strings = [strings]
        if isinstance(strings, list):
            strings = [[string] for string in strings]

        fields = [('text', self._TEXT)]

        list_of_examples = [Example.fromlist(string, fields) for string in strings]
        dataset = torchtext.data.Dataset(list_of_examples, fields)

        data = Iterator(dataset, batch_size=1, device=torch.device("cpu"), sort=False, sort_within_batch=False,
                        repeat=False, shuffle=False)

        predictions = []

        for item in data:
            x = item.text
            self.model.to(self.device)
            self.model = self.model.eval()
            outputs = self.model(x.to(self.device))

            predictions.extend(outputs.detach().cpu().argmax(1).tolist())
        results = [self._label_names[i] for i in predictions]
        return results

    def score(self, file_name: str, delimiter: str = ",", quotechar='"', text_col_name: str = 'text',
              label_col_name: str = 'label', batch_size: int = 64) -> None:
        """

        method that can be used score that model on an unseen test file

        :param file_name: string specifying the location and name of the file that contains the training dat
        :param delimiter: string specifying the delimiter used in the training csv file
        :param quotechar: string specifying the quotechar used in the training csv file
        :param text_col_name: string specifying the name of the column containing the mails in the csv file
        :param label_col_name: string specifying the name of the column containing the labels of the mails \
        in the csv file
        :param batch_size: integer specifying the batch size, this will affect the size of the batches fed into \
        the model this can be set lower if memory issues occur
        """
        assert self.has_trained
        print("Evaluating model")

        print("--- Starting with reading in the dataset ---")
        dataset_loader = CSVDataset(text_field=self._TEXT, file_name=file_name)
        dataset = dataset_loader.load(delimiter=delimiter, quotechar=quotechar, text_col_name=text_col_name,
                                      label_col_name=label_col_name)
        print("--- Finished with reading in the dataset ---")

        dloader = CustomDataLoader(dataset)
        data_iterator = dloader.construct_iterators(batch_size=batch_size, text_col_name=text_col_name,
                                                    label_col_name=label_col_name, is_test_set=True)

        generic_evaluation(self.model, data_iterator, self._criterion, device=self.device)
        return None

    def save_model(self, filename: str) -> None:
        """

        method that can be used to save a (trained) classifier using the '.pt' extension

        :param filename: string specifying the location and name of the destination of the saved model
        """
        assert filename.split(".")[-1] == "pt"
        torch.save(self.model.state_dict(), filename)
        return None

    def load_model(self, filename: str) -> None:
        """

        method that can be used to load a classifier saved in the .pt format

        :param filename: string specifying the name and location of the saved model to be loaded
        """
        assert filename.split(".")[-1] == "pt"
        self.model.load_state_dict(torch.load(filename))
        return None

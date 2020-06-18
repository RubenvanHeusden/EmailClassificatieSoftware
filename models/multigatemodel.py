"""
This class implements a wrapper around the main class of the Multigate Mixture of Experts model,
making it easier to train the model and evaluate new examples.
"""

import torch
import torchtext
import pandas as pd
import torch.nn as nn
from models.cnn import CNN
from models.mlp import MLP
import torch.optim as optim
from configurations import ROOT_DIR
from torch.optim.lr_scheduler import StepLR
from models.multitaskcnn import MultitaskConvNet
from torchtext.data import Field, Example, Iterator
from code_utils.multitaskcsvdataset import MultitaskCSVDataset
from code_utils.multitaskdataloader import MultitaskCustomDataLoader
from models.multigatemixtureofexperts import MultiGateMixtureofExperts
from code_utils.utils import embeddings_available, download_word_embeddings_nl, multitask_training,\
    multitask_evaluation, multitask_class_weighting
from collections import defaultdict


class MultigateModel:
    def __init__(self, num_outputs_list, target_names_list, n_experts: int = 3,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 word_embedding_path: str = ROOT_DIR+'/resources/word_embeddings/combined-320.tar/320/',
                 max_seq_len=None):
        """

        :param num_outputs_list: list specifying the number of class for each tasks separately, retrieved \
        using the 'get_num_labels_from_file' method from the utils file
        :param target_names_list: list of the name of the targets, this should be in the same order in which \
        the headers occur in the dataset
        :param n_experts: The number of experts the model uses, default is 3
        :param device: the device in which the inputs and model are set, this defaults to a GPU \
        when available
        :param word_embedding_path: string specifying the path to the word embeddings file
        :param max_seq_len: integer specifying maximum length that the sentences can be.
        """
        if not embeddings_available():
            download_word_embeddings_nl()
            print("--- Constructing the Pytorch embedding matrix file ---")
            torchtext.vocab.Vectors('combined-320.txt', cache=word_embedding_path)

        vocab_data = torch.load(word_embedding_path + "combined-320.txt.pt")

        self.device = device

        self._words, self._embed_dict, self._embeddings, self._embed_dim = vocab_data

        towers = {MLP(100 * 3, [128], output_dim): name for output_dim, name in zip(num_outputs_list,
                                                                                    target_names_list)}

        self._TEXT = Field(lower=True, tokenize="spacy", tokenizer_language="nl_core_news_sm", include_lengths=False,
                           batch_first=True, fix_length=max_seq_len)

        gating_networks = [CNN(input_channels=1, filter_list=(3, 4, 5),
                               embed_matrix=torch.zeros(size=(1, 1)), num_filters=25, output_dim=n_experts) for _ in
                           range(len(target_names_list))]

        shared_layers = [MultitaskConvNet(input_channels=1, filter_list=(3, 4, 5),
                                          embed_matrix=torch.zeros(size=(1, 1)), num_filters=100)
                         for _ in range(n_experts)]

        self.model = MultiGateMixtureofExperts(shared_layers=shared_layers, gating_networks=gating_networks,
                                          towers=towers, device=device, include_lens=False)

        self.num_outputs_list = num_outputs_list

        self.target_names_list = target_names_list

        self._criterion = None

        self._label_names = None

        self.has_trained = False

    def train_from_file(self, file_name, batch_size: int = 5, learning_rate: float = 0.1,
                        number_of_epochs: int = 10, delimiter: str = ",",
                        quotechar: str = '"'):
        """

        The main method of this class, implementing a training procedure for the model and handling
        the proper loading of the dataset

        :param file_name: string specifying the location and name of the file that contains the training dat
        :param batch_size: integer specifying the batch size, this will affect the size of the batches fed into \
        the model this can be set lower if memory issues occur
        :param number_of_epochs: integer specifying the number of epochs for which the model is trained. \
        The right amount of \
        epochs can differ for different datasets and it is recommended to inspect the produced TensorBoard logs \
        to see if the model has converged
        :param delimiter: string specifying the delimiter used in the training csv file
        :param quotechar: string specifying the quotechar used in the training csv file
        :param learning_rate: float specifying the learning rate of the model, this can affect the speed of \
        convergence of the model
        """
        print("--- Starting with reading in the dataset ---")
        dataset_loader = MultitaskCSVDataset(text_field=self._TEXT, path_to_datadir=file_name)
        dataset = dataset_loader.load( targets=self.target_names_list, delimiter=delimiter, quotechar=quotechar)
        print("--- Finished with reading in the dataset ---")

        dloader = MultitaskCustomDataLoader(dataset, self._TEXT, self.target_names_list)
        data_iterators = dloader.construct_iterators(batch_size=batch_size, device=torch.device("cpu"))

        self._TEXT.vocab.set_vectors(self._embed_dict, self._embeddings, self._embed_dim)
        for layer in self.model.shared_layers:
            layer.set_new_embedding_matrix(self._TEXT.vocab.vectors)
        for gate in self.model.gating_networks:
            gate.set_new_embedding_matrix(self._TEXT.vocab.vectors)

        self._label_names = {name: dataset.fields[name].vocab.itos for name in self.target_names_list}

        task_weights = multitask_class_weighting(data_iterators, self.target_names_list)
        losses = {name: nn.CrossEntropyLoss(weight=task_weights[name].to(self.device)) for name
                  in self.target_names_list}

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=1000, gamma=1.0)

        multitask_training(self.model, losses, optimizer, scheduler, data_iterators, device=self.device,
                           include_lengths=False, save_path=ROOT_DIR+'/runs/',
                           save_name="custom_dataset",
                           tensorboard_dir=ROOT_DIR+"/runs/", n_epochs=number_of_epochs,
                           checkpoint_interval=100)
        self.has_trained = True
        return None

    def classify_from_file(self, file_name, batch_size: int = 5, delimiter: str = ",", quotechar: str = '"',
                           text_col_name: str = 'text'):
        assert self.has_trained

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

        predictions = defaultdict(list)

        for item in data:
            x = getattr(item, text_col_name)
            # Set the model to evaluation mode, important because of the Dropout Layers
            self.model.to(self.device)
            self.model = self.model.eval()
            outputs = self.model(x.to(self.device), tower=self.target_names_list)
            for i in range(len(self.target_names_list)):
                predictions[self.target_names_list[i]].extend(outputs[i].detach().cpu().argmax(1).tolist())

        results = defaultdict(list)
        for key, val in predictions.items():
            results[key] = [self._label_names[key][i] for i in predictions[key]]
        return results

    def classify_from_strings(self, strings):
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

        predictions = defaultdict(list)

        for item in data:
            x = item.text
            # Set the model to evaluation mode, important because of the Dropout Layers
            self.model.to(self.device)
            self.model = self.model.eval()
            outputs = self.model(x.to(self.device), tower=self.target_names_list)
            for i in range(len(self.target_names_list)):
                predictions[self.target_names_list[i]].extend(outputs[i].detach().cpu().argmax(1).tolist())

        results = defaultdict(list)
        for key, val in predictions.items():
            results[key] = [self._label_names[key][i] for i in predictions[key]]
        return results

    def score(self, file_name, batch_size: int = 5, delimiter: str = ",", quotechar: str = '"'):
        """

        method that can be used score that model on an unseen test file

        :param file_name: string specifying the location and name of the file that contains the training dat
        :param delimiter: string specifying the delimiter used in the training csv file
        :param quotechar: string specifying the quotechar used in the training csv file \
        in the csv file
        :param batch_size: integer specifying the batch size, this will affect the size of the batches fed into \
        the model this can be set lower if memory issues occur
        """
        assert self.has_trained

        self.model.to(self.device)
        self.model = self.model.eval()

        print("--- Starting with reading in the dataset ---")
        dataset_loader = MultitaskCSVDataset(text_field=self._TEXT, path_to_datadir=file_name)
        dataset = dataset_loader.load(targets=self.target_names_list, delimiter=delimiter, quotechar=quotechar)
        print("--- Finished with reading in the dataset ---")
        dloader = MultitaskCustomDataLoader(dataset, self._TEXT, self.target_names_list)
        data_iterators = dloader.construct_iterators(batch_size=batch_size, device=torch.device("cpu"),
                                                     is_test_set=True)

        multitask_evaluation(self.model, data_iterators, device=self.device)

    def save_model(self, filename: str) -> None:
        """

        method that can be used to save a (trained) classifier

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


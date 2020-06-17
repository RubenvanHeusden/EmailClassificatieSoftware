"""
This file contains the main cnn model that is used by the CNNClassifier method, where the Classifier file
takes care of converting the inputs of the model to the right format
"""

import torch
import torch.nn as nn
from typing import List, Union, Tuple


class CNN(nn.Module):
    """
    This class implements a basic convolutional neural network for
    text classification as presented in (Kim,. 2014)

    """
    def __init__(self, input_channels: int, output_dim: int, filter_list: Union[List[int], Tuple[int]],
                 embed_matrix, num_filters: int, dropbout_probs: float = 0.5):
        """

        :param input_channels: integer specifying the number of input channels of the 'image'
        :param output_dim: integer specifying the number of outputs
        :param filter_list: list of integers specifying the size of each kernel that is applied to the image, \
        the outputs of the kernels are concatenated to form the input to the linear layer
        :param embed_matrix: torch Tensor with size [size_vocab, embedding_dim] where each row is a \
        word embedding
        :param num_filters: the amount of filter to apply to the image, this also determines the size \
        of the input to the fully connected layers, which is equal to num_kernels*num_filters
        :param dropbout_probs: float specifying the dropout probabilities
        """
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.filters = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters,
                                                kernel_size=(n, embed_matrix.shape[1])) for n in filter_list])

        self.max_over_time_pool = torch.nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.fc_layer = nn.Linear(num_filters*len(filter_list), output_dim)
        self.dropout = nn.Dropout(p=dropbout_probs)
        self.embed = nn.Embedding(*embed_matrix.shape)
        self.embed.weight.data.copy_(embed_matrix)
        self.embed.requires_grad = True
        self.relu = nn.ReLU()
        self.num_filters = num_filters
        self.filter_list = filter_list

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        :param x: input of size (batch_size, max_sen_length_batch) where rows are the sentences and
        each entry is the index of the word into the word matrix
        :return: output of the CNN applied to the input x
        """

        b = x.shape[0]

        x = x.unsqueeze(1)
        x = self.embed(x)
        filter_outs = []
        for module in self.filters:
            module_out = self.relu(module(x))
            module_out = self.max_over_time_pool(module_out)
            filter_outs.append(module_out)
        pen_ultimate_layer = torch.cat(filter_outs, dim=1)
        output = self.dropout(pen_ultimate_layer).squeeze()
        output = self.fc_layer(output)

        return output.view(b, -1)

    def set_new_embedding_matrix(self, embedding_matrix: torch.FloatTensor) -> None:
        """
        :param embedding_matrix: torch.FloatTensor of shape [num_words_in_vocab, embedding_dim] specifying the
        new matrix that should be used as embeddding matrix
        :return: None
        """
        self.embed = nn.Embedding(*embedding_matrix.shape)
        self.embed.weight.data.copy_(embedding_matrix)
        self.filters = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                                                kernel_size=(n, embedding_matrix.shape[1])) for n in self.filter_list])
        return None

"""
This file contains an implementation of a CNN model that is suited for use in a Multigate model.
This only difference between this model and the other CNN model included in this package is that this one
this not have a separate final layer and supports the feeding of multiple tasks simultaneously to the model.
"""

import torch
import torch.nn as nn


class MultitaskConvNet(nn.Module):
    """
    This class implements a basic convolutional neural network for
    text classification as presented in (Kim,. 2014)

    Because this model is used in the multitask setting, the output of the model is not
    fed through a linear layer to accomodate for different task output size.
    """
    def __init__(self, input_channels, filter_list, embed_matrix, num_filters, dropbout_probs=0.5):
        super(MultitaskConvNet, self).__init__()
        self.params = locals()
        self.input_channels = input_channels
        self.filters = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters,
                                                kernel_size=(n, embed_matrix.shape[1])) for n in filter_list])

        self.max_over_time_pool = torch.nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.dropout = nn.Dropout(p=dropbout_probs)
        self.embed = nn.Embedding(*embed_matrix.shape)
        self.embed.weight.data.copy_(embed_matrix)
        self.num_filters = num_filters
        self.filter_list = filter_list
        self.relu = nn.ReLU()

    def forward(self, x):
        b = x.shape[0]
        x = x.unsqueeze(1)
        x = self.embed(x)
        filter_outs = []
        for module in self.filters:
            filter_outs.append(self.max_over_time_pool(self.relu(module(x))))
        pen_ultimate_layer = torch.cat(filter_outs, dim=1)
        output = self.dropout(pen_ultimate_layer).squeeze()

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




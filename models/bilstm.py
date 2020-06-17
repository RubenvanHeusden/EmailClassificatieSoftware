"""
This file contains the implementation of a Bidirectional LSTM classifier implementation from PyTorch.
"""


import torch
import torch.nn as nn
from typing import List


class BiLSTM(nn.Module):
    """
    This class contains the implementation of a Bidirectional LSTM model from PyTorch

    Attributes
    ----------
    embed : nn.Embedding
    torch.nn.Embedding class that holds the embeddings of the words in the vocabulary

    embedding_dim: int
        integer specifying the dimension of the word embeddings

    hidden_dim: int
        Integer specifying the size of the hidden layer in the LSTM network, this is the size of the
        hidden layer for each direction separately, so the total number of parameters in the model
        is double the size of hidden_dim

    output_dim: int
        Integer specifying the number of outputs of the model, this should be equal to the number of classes
        in the dataset

    lstm: nn.LSTM
        the nn.LSTM model from pytorch that is used as the main part of this implementation

    fc_out: int
        Integer specifying the number of neurons in the linear layer that follows the LSTM models

    dropout: float
        Float specifying the amount of dropout applied to the penultimate linear layer of the model

    device: torch.device
        torch.device specifying the device on which the model and the inputs to the model are set,
        the default behaviour is to check for the availability of a GPU on the system and use this if
        its is available

    """
    def __init__(self, vocab: torch.Tensor, hidden_dim: int, output_dim: int, dropout: float = 0.3,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        :param vocab: torch.Tensor of size [num_words_in_vocab * word_embedding_dim] where the row indices
        in the matrix should correspond to the word embedding vectors
        :param hidden_dim: integer specifying the hidden dimension of the Bidirectional model
        :param output_dim: integer specifying the output dimension of the network, this corresponds to the
        number of possible classes any data-point can have
        :param dropout: float specifying the amount of dropout that is used in the pen-ultimate linear
        layer, default is 0.3
        :param device: torch.device specifying the device on which the inputs and the model should be put.
        By default the model will be put on the GPU if one is available
        """
        super(BiLSTM, self).__init__()
        self.embed = nn.Embedding(*vocab.shape)
        self.embed.weight.data.copy_(vocab)
        self.embedding_dim = vocab.shape[1]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_out = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.to(device)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """

        the forward method is responsible for passing an input to the network and calculating the output.
        This method automatically incorporates the lengths of the inputs passed to it as to minimize the
        padding needed.

        :param x: a list with contents [batch_word_embeddings_indices, vector_with_sentence_lengths]
        to be fed into the model
        :return: the outpus of the model of shape [num_outputs, 1]
        """
        inputs, lengths = x
        b = inputs.shape[0]
        inputs = self.embed(inputs)
        x = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

        h_0 = torch.zeros(2, b, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(2, b, self.hidden_dim).to(self.device)

        torch.nn.init.xavier_normal_(h_0)
        torch.nn.init.xavier_normal_(c_0)

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        final_hidden_state = torch.cat([final_hidden_state[0, :, :], final_hidden_state[1, :, :]], dim=1)

        return self.fc_out(self.dropout(final_hidden_state))

    def set_new_embedding_matrix(self, embedding_matrix: torch.FloatTensor) -> None:
        """

        this method can be used to set a new embedding matrix, it automatically changes the parameters of the
        model that are affected by this change

        :param embedding_matrix: torch.FloatTensor of shape [num_words_in_vocab, embedding_dim] specifying the
        new matrix that should be used as embeddding matrix
        :return: None
        """
        self.embed = nn.Embedding(*embedding_matrix.shape)
        self.embed.weight.data.copy_(embedding_matrix)
        self.embedding_dim = embedding_matrix.shape[1]
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        return None

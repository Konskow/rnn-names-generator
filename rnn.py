import torch

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.hidden_size)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.n_layers, dropout=0.2)
        self.decoder = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.LogSoftmax()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size), torch.zeros(self.n_layers, 1, self.hidden_size)

    def forward(self, input):
        input = self.encoder(input.view(1, -1))
        output, self.hidden = self.rnn(input, self.hidden)
        output = self.decoder(output.view(1, -1))

        return output

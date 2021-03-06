'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn

class FS_MODEL1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(FS_MODEL1, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

class FS_MODEL2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(FS_MODEL2, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, input, hidden):
        
        output, hidden = self.rnn(input, hidden)        
        output = self.fc(output)
        output = self.softmax(output)

        return output, hidden

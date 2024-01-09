import torch
import torch.nn as nn

"""
    input : [batch_size, input_size(O=96), seq_len]
    output: [batch_size, output_size(O=96/O=336), seq_len]
"""

class TimeSeriesCNN_96(nn.Module):
    def __init__(self, input_size=96, output_size=96, seq_length=7, hidden_size=64, lstm_layer=3, dropout=0.1):
        super(TimeSeriesCNN_96, self).__init__()

        self.lstm = nn.LSTM(input_size=seq_length,
                            hidden_size=hidden_size,
                            num_layers=lstm_layer,
                            batch_first=True)

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2) # [batch_size, 128, 16]

        self.Linear1 = nn.Linear(128*16, output_size*(seq_length))

        self.output_size_1 = output_size
        self.output_size_2 = seq_length

    def forward(self, x):
        batch_size = x.shape[0]
        lstm_out, (h_n, c_n) = self.lstm(x) 
        x = self.conv1(lstm_out)            
        x = self.bn1(x)
        x = self.conv2(x)                   

        x = x.view(-1, 128*16)
        x = self.Linear1(x)

        x = x.view(-1, self.output_size_1, self.output_size_2)
        return x

class TimeSeriesCNN_336(nn.Module):
    def __init__(self, input_size=96, output_size=336, seq_length=7, hidden_size=256, lstm_layer=5, dropout=0.1):
        super(TimeSeriesCNN_336, self).__init__()

        self.lstm = nn.LSTM(input_size=seq_length,
                            hidden_size=hidden_size,
                            num_layers=lstm_layer,
                            batch_first=True)

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm1d(num_features=64)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=2, stride=2) # [batch_size, 128, 16]

        self.Linear1 = nn.Linear(128*64, output_size*(seq_length))

        self.output_size_1 = output_size
        self.output_size_2 = seq_length

    def forward(self, x):
        batch_size = x.shape[0]
        lstm_out, (h_n, c_n) = self.lstm(x) 
        x = self.conv1(lstm_out)            
        x = self.bn1(x)
        x = self.conv2(x)                  

        x = x.view(-1, 128*64)
        x = self.Linear1(x)

        x = x.view(-1, self.output_size_1, self.output_size_2)
        return x

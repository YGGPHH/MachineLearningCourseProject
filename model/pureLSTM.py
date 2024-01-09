import torch
import torch.nn as nn
import torch.nn.functional as F

class pureLSTM_96(nn.Module):
    def __init__(self, input_size=96, output_size=96, seq_length=7, hidden_size=128, lstm_layer=3, dropout=0.2):
        super(pureLSTM_96, self).__init__()
        self.output_size = output_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = lstm_layer

        self.lstm = nn.LSTM(input_size=seq_length,
                            hidden_size=hidden_size,
                            num_layers=lstm_layer,
                            batch_first=True)

        self.Linear_2 = nn.Linear(input_size * hidden_size, output_size * seq_length)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.shape[0]
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.reshape(batch_size, -1)

        x = self.dropout(lstm_out)
        x = self.Linear_2(x)

        x = x.view(-1, self.output_size, self.seq_length)
        return x
    
class pureLSTM_336(nn.Module):
    def __init__(self, input_size=96, output_size=336, seq_length=7, hidden_size=256, lstm_layer=5, dropout=0.2):
        super(pureLSTM_336, self).__init__()
        self.lstm = nn.LSTM(input_size=seq_length,
                                  hidden_size=hidden_size,
                                  num_layers=lstm_layer,
                                  batch_first=True)

        self.Linear_1 = nn.Linear(input_size * hidden_size, output_size * 32)
        self.Linear_2 = nn.Linear(output_size * 32, output_size * seq_length)
        self.dropout = nn.Dropout(p=dropout)
        
        self.output_size_1 = output_size
        self.output_size_2 = seq_length

    def forward(self, x):
        batch_size = x.shape[0]
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = lstm_out.reshape(batch_size, -1)
        
        x = self.dropout(lstm_out)
        x = self.Linear_1(x)
        x = self.Linear_2(x)
        x = x.view(-1, self.output_size_1, self.output_size_2)
        return x

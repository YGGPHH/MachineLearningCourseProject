import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Transformer_96(nn.Module):
    def __init__(self, input_size=96, output_size=96, seq_len=7, d_model=128):
        super(Transformer_96, self).__init__()

        self.input_fc = nn.Linear(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.positional_embedding = PositionalEncoding(d_model, dropout=0.1)
        self.fc1 = nn.Linear(input_size * d_model, d_model)
        self.fc2 = nn.Linear(d_model, output_size*(seq_len))
        self.output_size_1 = output_size
        self.output_size_2 = seq_len

    def forward(self, x):
        # x.shape = [batch_size, input_size, seq_len]
        x = self.input_fc(x)  # [batch_size, input_size, embedding_dims]
        x = self.positional_embedding(x)
        x = self.encoder(x)
        # x.shape = [batch_size, input_size, d_model]
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, self.output_size_1, self.output_size_2)
        return x

class Transformer_336(nn.Module):
    def __init__(self, input_size=96, output_size=336, seq_len=7, d_model=256):
        super(Transformer_336, self).__init__()

        self.input_fc = nn.Linear(seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.positional_embedding = PositionalEncoding(d_model, dropout=0.1)
        self.fc1 = nn.Linear(input_size * d_model, d_model)
        self.fc2 = nn.Linear(d_model, output_size*(seq_len))
        self.output_size_1 = output_size
        self.output_size_2 = seq_len

    def forward(self, x):
        x = self.input_fc(x)  # [batch_size, input_size, embedding_dims]
        x = self.positional_embedding(x)
        x = self.encoder(x)
        # x.shape = [batch_size, input_size, d_model]
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, self.output_size_1, self.output_size_2)
        return x
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config["vocab_size"], config["embed_size"], padding_idx=0)
        self.lstm = nn.LSTM(
            config["embed_size"],
            config["hidden_size"],
            batch_first=True)
        self.config = config

    def forward(self, x, x_l):
        embed = pack_padded_sequence(self.embedding(x), x_l, batch_first=True) 
        return self.lstm(embed)
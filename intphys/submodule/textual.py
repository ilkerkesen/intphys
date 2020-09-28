import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config["input_size"], config["embed_size"], padding_idx=0)
        self.lstm = nn.LSTM(config["embed_size"], config["hidden_size"])
        self.config = config

    def forward(self, x):
        return self.lstm(self.embedding(x))

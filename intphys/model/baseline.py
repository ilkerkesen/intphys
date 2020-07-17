import torch
import torch.nn as nn


class BlindBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config["input_size"],
                                      self.config["embed_size"],
                                      padding_idx=0)
        self.lstm = nn.LSTM(config["embed_size"], config["hidden_size"])
        self.softmax = nn.Linear(config["hidden_size"], config["output_size"])

    def forward(self, x):
        embeddings = self.embedding(x)
        _, (hiddens, _) = self.lstm(embeddings)
        answers = self.softmax(hiddens.squeeze(0))
        return answers

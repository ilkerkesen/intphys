import torch
import torch.nn as nn

class BlindBaseline(nn.Module):
    def __init__(self, config):
        self.config = config
        self.lstm = nn.LSTM(config["input_size"], config["hidden_size"])
        self.softmax

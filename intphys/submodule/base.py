import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential()
        func = config["activation"].lower()
        nonlinear = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh
        }[func]
        for i in range(config["num_layers"]):
            out_features = config["hidden_size"]
            in_features = config["input_size"] if i == 0 else out_features
            linear = nn.Linear(
                in_features=in_features,
                out_features=out_features
            )
            self.layers.add_module("dense{}".format(i), linear)
            self.layers.add_module("{}{}".format(func, i), nonlinear())

    def forward(self, x):
        return self.layers(x)

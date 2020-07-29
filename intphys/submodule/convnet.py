import torch
import torch.nn as nn

from .base import MLP


class ShallowCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential()
        conv_w = config.get("input_width", 256)
        conv_h = config.get("input_height", 256)
        kernel_size, stride = config["kernel_size"], config["stride"]
        for i in range(config["num_layers"]):
            out_channels = (2**i) * config["num_channels"]
            in_channels = 3 if i == 0 else out_channels // 2
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride)
            bn = nn.BatchNorm2d(out_channels)

            self.layers.add_module("conv{}".format(i), conv)
            self.layers.add_module("bn{}".format(i), bn)
            self.layers.add_module("relu{}".format(i), nn.ReLU())

            conv_w = (conv_w - (kernel_size-1) - 1) // stride  + 1
            conv_h = (conv_h - (kernel_size-1) - 1) // stride  + 1

        self.out_features = conv_w * conv_h * out_channels

        # mlp_config = config["mlp"]
        # mlp_config["mlp"]["in_features"] = in_features
        # mlp = MLP(config)
        # self.layers.add_module("mlp", mlp)

    def forward(self, x):
        return self.layers(x)

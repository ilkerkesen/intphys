from copy import copy


import torch
import torch.nn as nn
from torchvision.models import resnet18 as _resnet18
from torchvision.transforms import Compose, Normalize, Lambda


class CNN2Dv1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.normalizer = None
        self.setup_layers()

    def forward(self, x):
        return self.layers(x)

    def setup_layers(self):
        self.layers = nn.Sequential()
        size = self.config["input_size"]
        for idx in range(self.config["num_layers"]):
            size = self.calculate_output_size(size)
            self.add_layer(idx=idx)
        num_channels = (2**idx) * self.config["num_channels"]
        self.out_features = size**2 * num_channels

    def add_layer(self, idx):
        out_channels = (2**idx) * self.config["num_channels"]
        in_channels = 3 if idx == 0 else out_channels // 2

        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.config["kernel_size"],
            stride=self.config["stride"],
            padding=self.config["padding"])
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()

        self.layers.add_module("conv{}".format(idx), conv)
        self.layers.add_module("bn{}".format(idx), bn)
        self.layers.add_module("relu{}".format(idx), relu)

    def calculate_output_size(self, input_size):
        W = self.config["kernel_size"]
        P = self.config["padding"]
        S = self.config["stride"]
        return (input_size + 2*P - W) // S + 1


def resnet18(config):
    net = _resnet18(pretrained=config["pretrained"], progress=True)
    layers = list(net.children())
    net = nn.Sequential(*layers[:4+config["num_layers"]-1])
    if config["pretrained"] and config["freeze"]:
        net.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for par in net.parameters():
            par.requires_grad = False
    out_size = config["input_size"] // 4 // 2**(config["num_layers"]-2)
    out_channels = 2**(5+config["num_layers"]-1)
    net.out_features = out_channels * out_size * out_size
    net.config = config
    return net

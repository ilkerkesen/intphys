import torch
import torch.nn as nn


from .base import MLP


def create_cbr_module(in_channels, out_channels, kernel_size, stride, depth):
    conv_module, bn_module = nn.Conv2d, nn.BatchNorm2d
    if depth: conv_module, bn_module = nn.Conv3d, nn.BatchNorm3d
    conv = conv_module(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride)
    bn = bn_module(out_channels)
    relu = nn.ReLU()
    return (conv, bn, relu)


def calculate_output_shape(width, height, kernel_size, stride, depth=0):
    assert type(kernel_size) == type(stride)
    assert (depth > 0 and len(kernel_size) == len(stride) == 3) or True
    assert isinstance(depth, int)

    if isinstance(kernel_size, int):
        kernel_width = kernel_height = kernel_size
        stride_width = stride_height = stride
    elif isinstance(kernel_size, list):
        kernel_width, kernel_height = kernel_size[-2], kernel_size[-1]
        stride_width, stride_height = stride[-2], stride[-1]

    calculate_new = lambda old, k, s: (old - (k-1)) // s + 1
    new_width = calculate_new(width, kernel_width, stride_width)
    new_height = calculate_new(height, kernel_height, stride_height)
    new_depth = depth
    if depth > 0:
        kernel_depth, kernel_stride = kernel_size[0], stride[0]
        new_depth = calculate_new(depth, kernel_depth, kernel_stride)

    return (new_width, new_height, new_depth)


class ShallowCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential()
        conv_w = config.get("input_width", 256)
        conv_h = config.get("input_height", 256)
        conv_d = config.get("input_depth", 0)
        depth = conv_d > 0
        kernel_size = [k for k in config["kernel_size"]]
        stride = [s for s in config["stride"]]
        if not depth:
            kernel_size = kernel_size[-2:]
            stride = stride[-2:]
        for i in range(config["num_layers"]):
            out_channels = (2**i) * config["num_channels"]
            in_channels = 3 if i == 0 else out_channels // 2
            conv, bn, relu = create_cbr_module(
                in_channels, out_channels, kernel_size, stride, depth=depth)

            self.layers.add_module("conv{}".format(i), conv)
            self.layers.add_module("bn{}".format(i), bn)
            self.layers.add_module("relu{}".format(i), relu)
            conv_w, conv_h, conv_d = calculate_output_shape(
                conv_w, conv_h, kernel_size, stride, depth=conv_d)
        self.out_features = conv_w * conv_h * out_channels
        self.out_features *= conv_d if depth else 1

    def forward(self, x):
        return self.layers(x)

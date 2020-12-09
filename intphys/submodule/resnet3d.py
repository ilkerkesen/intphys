import torch
import torch.nn as nn
from torchvision.models.video import r3d_18 as _r3d_18
from torchvision.transforms import Compose, Normalize, Lambda


def r3d_18(config):
    net = _r3d_18(pretrained=config["pretrained"], progress=True)
    layers = list(net.children())
    net = nn.Sequential(*layers[:config["num_layers"]])
    if config["pretrained"] and config["freeze"]:
        for par in net.parameters():
            par.requires_grad = False
    out_size = config["input_size"] // 2**(config["num_layers"]-1)
    out_channels = 2**(4+config["num_layers"])
    out_depth = config["depth_size"]
    for i in range(config["num_layers"]-2):
        out_depth = (out_depth+1) // 2
    net.out_features = out_depth * out_channels * out_size * out_size
    net.config = config
    return net

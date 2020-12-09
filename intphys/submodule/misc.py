import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FiLMLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.gamma_layer = nn.Linear(in_features, out_features)
        self.beta_layer = nn.Linear(in_features, out_features)

    def forward(self, x, c):
        gamma = self.gamma_layer(c).view(x.size(0), x.size(1), 1, 1)
        beta = self.beta_layer(c).view(x.size(0), x.size(1), 1, 1)
        return gamma * x + beta


class FiLMBlock2D(nn.Module):
    def __init__(self, in_features, in_channels, out_channels,
        kernel_size, stride, padding):
        super(FiLMBlock2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.film = FiLMLayer(
            in_features=in_features,
            out_features=out_channels
        )

    def forward(self, x, c):
        x = self.conv(x)
        x = F.relu(x)
        x = self.bnorm(x)
        x = self.film(x, c)
        return x
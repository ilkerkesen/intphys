import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..submodule import *
from ..data import SimulationInput


__all__ = (
    "FiLM2D",
    "FiLM2DFF",
    "FiLM2DLF",
)


class FiLM2D(nn.Module):
    SIMULATION_INPUT = SimulationInput.NO_FRAMES
    NUM_VIDEO_FRAMES = 0

    """Some Information about FiLM2D"""
    def __init__(self, config):
        super(FiLM2D, self).__init__()
        config["question_encoder"]["vocab_size"] = config["input_size"]
        self.question_encoder = self.create_submodule(config, "question_encoder")

        # initialize FiLM network
        size = config["frame_encoder"]["input_size"]
        self.film_blocks = nn.ModuleList()
        for i in range(config["frame_encoder"]["num_layers"]):
            out_channels = 2**i * config["frame_encoder"]["num_channels"]
            in_channels = 3 if i == 0 else out_channels // 2
            kernel_size = config["frame_encoder"]["kernel_size"]
            stride = config["frame_encoder"]["stride"]
            padding = config["frame_encoder"]["padding"]
            
            block = FiLMBlock2D(
                in_features=config["question_encoder"]["hidden_size"],
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )

            self.film_blocks.append(block)
            size = (size + 2 * padding - kernel_size) // stride + 1

        out_features = size**2 * out_channels
        vis_features = self.NUM_VIDEO_FRAMES * out_features
        txt_features = config["question_encoder"]["hidden_size"]
        config["mlp"]["input_size"] = vis_features + txt_features
        self.mlp = MLP(config["mlp"])
        self.linear = nn.Linear(
            in_features=config["mlp"]["hidden_size"], out_features=config["output_size"]
        )
        self.config = config

    def create_submodule(self, config, submodule):
        _config = config[submodule]
        submodule = eval(_config["architecture"])(_config)
        return submodule

    def process_question(self, questions, **kwargs):
        _, (hiddens, _) = self.question_encoder(questions)
        return hiddens.squeeze(0)

    def process_simulation(self, simulations, features, **kwargs):
        x = simulations
        for block in self.film_blocks:
            x = block(x, features)
        return x

    def forward(self, simulations, questions, **kwargs):
        txt = self.process_question(questions, **kwargs) 
        vis = self.process_simulation(simulations, txt, **kwargs)
        y = torch.cat([torch.flatten(vis, start_dim=1), txt], dim=1)
        y = self.mlp(y)
        return self.linear(y)


class FiLM2DFF(FiLM2D):
    SIMULATION_INPUT = SimulationInput.FIRST_FRAME
    NUM_VIDEO_FRAMES = 1


class FiLM2DLF(FiLM2D):
    SIMULATION_INPUT = SimulationInput.LAST_FRAME
    NUM_VIDEO_FRAMES = 1
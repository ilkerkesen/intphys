import torch
import torch.nn as nn


from ..submodule import *
from ..data import SimulationInput


class LRCNBaseline(nn.Module):
    SIMULATION_INPUT = SimulationInput.VIDEO

    def __init__(self, config):
        super().__init__()
        config["question_encoder"]["vocab_size"] = config["input_size"]
        self.config = config
        self.frame_encoder = self.create_submodule("frame_encoder")
        self.question_encoder = self.create_submodule("question_encoder")
        num_frame_features = self.frame_encoder.out_features
        num_question_features = self.question_encoder.config["hidden_size"]
        num_video_features = config["video_encoder"]["hidden_size"]
        config["video_encoder"]["input_size"] = num_video_features
        self.video_encoder = nn.LSTM(**config["video_encoder"])
        self.frame_projector = nn.Linear(
            in_features=num_frame_features,
            out_features=num_video_features)
        self.flatten = nn.Flatten()
        config["mlp"]["input_size"] = num_video_features + num_question_features
        self.mlp = MLP(config["mlp"])
        self.linear = nn.Linear(
            in_features=config["mlp"]["hidden_size"],
            out_features=config["output_size"])

    def create_submodule(self, submodule):
        config = self.config[submodule]
        submodule = eval(config["architecture"])(config)
        return submodule

    def process_simulation(self, simulations, **kwargs):
        # B, C, D, W, H = simulations.shape
        simulations = torch.transpose(simulations, 1, 2)
        B, D, C, W, H = simulations.shape
        simulations = simulations.reshape(B*D, C, W, H)
        y = self.frame_encoder(simulations)
        y = self.flatten(y)
        y = self.frame_projector(y)
        y = y.reshape(B, D, -1).transpose(0, 1)
        _, (hiddens, _) = self.video_encoder(y)
        y = hiddens.squeeze(0)
        return y

    def process_question(self, questions, **kwargs):
        _, (hiddens, _) = self.question_encoder(questions, **kwargs)
        return hiddens.squeeze(0)

    def forward(self, simulations, questions, **kwargs):
        vis = self.process_simulation(simulations, **kwargs)
        txt = self.process_question(questions, **kwargs)
        y = torch.cat([vis, txt], dim=1)
        y = self.mlp(y)
        return self.linear(y)

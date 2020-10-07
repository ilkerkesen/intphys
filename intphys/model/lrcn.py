import torch
import torch.nn as nn


from ..submodule import *
from ..data import SimulationInput


class LRCNBaseline(nn.Module):
    SIMULATION_INPUT = SimulationInput.VIDEO

    def __init__(self, config):
        super().__init__()
        # FIXME: delete these lines
        if config["question_encoder"]["input_size"] is None:
            config["question_encoder"]["input_size"] = config.pop("input_size")
        if config["num_answers"] is None:
            config["num_answers"] = config.pop("output_size")

        self.text_encoder = LSTMEncoder(config["question_encoder"])
        self.convnet = eval(config["convnet"]["architecture"])(config["convnet"])
        self.flatten = nn.Flatten()
        config["mlp"]["input_size"] = self.convnet.out_features
        config["video_encoder"]["input_size"] = config["mlp"]["hidden_size"]
        self.mlp = MLP(config["mlp"])
        self.video_encoder = nn.LSTM(**config["video_encoder"])
        visual_features_dim = config["video_encoder"]["hidden_size"]
        question_features_dim = config["question_encoder"]["hidden_size"]
        self.linear = nn.Linear(
            in_features=visual_features_dim + question_features_dim,
            out_features=config["num_answers"])
        self.config = config
        self.NUM_VIDEO_FRAMES = self.config.get("num_frames", 10)

    def process_simulation(self, simulations, **kwargs):
        # B, C, D, W, H = simulations.shape
        simulations = torch.transpose(simulations, 1, 2)
        B, D, C, W, H = simulations.shape
        simulations = simulations.reshape(B*D, C, W, H)
        y = self.convnet(simulations)
        y = self.flatten(y)
        y = self.mlp(y)
        y = y.reshape(B, D, -1).transpose(0, 1)
        _, (hiddens, _) = self.video_encoder(y)
        y = hiddens.squeeze(0)
        return y

    def process_question(self, questions, **kwargs):
        _, (hiddens, _) = self.text_encoder(questions, **kwargs)
        return hiddens.squeeze(0)

    def forward(self, simulations, questions, **kwargs):
        vis = self.process_simulation(simulations, **kwargs)
        txt = self.process_question(questions, **kwargs)
        return self.linear(torch.cat([vis, txt], dim=1))

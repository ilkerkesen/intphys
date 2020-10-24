import torch
import torch.nn as nn

from ..submodule import *
from ..data import SimulationInput


__all__ = (
    "LSTMBaseline",
    "LSTMCNNBaselineFF",
    "LSTMCNNBaselineLF",
    "LSTMCNNBaseline2F",
)


class LSTMBaseline(nn.Module):
    """
    Does not use any kind of visual data.
    """
    SIMULATION_INPUT = SimulationInput.NO_FRAMES
    NUM_VIDEO_FRAMES = 0

    def __init__(self, config):
        super().__init__()
        config["question_encoder"]["vocab_size"] = config["input_size"]
        self.question_encoder = LSTMEncoder(config["question_encoder"])
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])
        self.dropout = nn.Dropout(p=config["dropout"])
        self.config = config

    def forward(self, simulations, questions, **kwargs):
        _, (hiddens, _) = self.question_encoder(questions)
        answers = self.linear(self.dropout(hiddens.squeeze(0)))
        return answers


class LSTMCNNBaseline(nn.Module):
    SIMULATION_INPUT = SimulationInput.NO_FRAMES
    NUM_VIDEO_FRAMES = 0

    def __init__(self, config):
        super().__init__()
        config["question_encoder"]["vocab_size"] = config["input_size"]
        self.config = config
        self.frame_encoder = self.create_submodule("frame_encoder")
        self.question_encoder = self.create_submodule("question_encoder")
        visual_size = self.NUM_VIDEO_FRAMES * self.frame_encoder.out_features
        textual_size = self.question_encoder.config["hidden_size"]
        config["mlp"]["input_size"] = visual_size + textual_size
        self.flatten = nn.Flatten()
        self.mlp = MLP(config["mlp"])
        self.linear = nn.Linear(
            in_features=config["mlp"]["hidden_size"],
            out_features=config["output_size"])
        self.dropout = nn.Dropout(p=config["dropout"])
        self.config = config

    def create_submodule(self, submodule):
        config = self.config[submodule]
        submodule = eval(config["architecture"])(config)
        return submodule

    def process_simulation(self, simulations, **kwargs):
        y = self.frame_encoder(simulations)
        y = self.flatten(y)
        return y

    def process_question(self, questions, **kwargs):
        _, (hiddens, _) = self.question_encoder(questions)
        return hiddens.squeeze(0)

    def forward(self, simulations, questions, **kwargs):
        vis = self.process_simulation(simulations, **kwargs)
        txt = self.process_question(questions, **kwargs)
        y = torch.cat([self.dropout(vis), self.dropout(txt)], dim=1)
        y = self.mlp(y)
        return self.linear(y)


class LSTMCNNBaselineFF(LSTMCNNBaseline):
    SIMULATION_INPUT = SimulationInput.FIRST_FRAME
    NUM_VIDEO_FRAMES = 1


class LSTMCNNBaselineLF(LSTMCNNBaseline):
    SIMULATION_INPUT = SimulationInput.LAST_FRAME
    NUM_VIDEO_FRAMES = 1


class LSTMCNNBaseline2F(LSTMCNNBaseline):
    SIMULATION_INPUT = SimulationInput.FIRST_AND_LAST_FRAMES
    NUM_VIDEO_FRAMES = 2

    def process_simulation(self, simulations, **kwargs):
        y = self.convnet(simulations)
        y = self.flatten(y)
        batch_size = y.size(0) // 2
        first_frames, last_frames = y[:batch_size], y[batch_size:]
        y = torch.cat([first_frames, last_frames], dim=1)
        return y

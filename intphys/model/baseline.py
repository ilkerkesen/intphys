import torch
import torch.nn as nn

from ..submodule import *
from ..data import SimulationInput


__all__ = (
    "TextualBaseline",
    "VisualBaseline",
    "FirstFrameVisualBaseline",
    "LastFrameVisualBaseline",
    "DoubleFramesVisualBaseline",
    "FirstFrameBaseline",
    "LastFrameBaseline",
    "DoubleFramesBaseline",
    "VideoBaseline",
)


class TextualBaseline(nn.Module):
    """
    Does not use any kind of visual data.
    """
    SIMULATION_INPUT = SimulationInput.NO_FRAMES
    NUM_VIDEO_FRAMES = 0

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config["input_size"],
                                      self.config["embed_size"],
                                      padding_idx=0)
        self.lstm = nn.LSTM(config["embed_size"], config["hidden_size"])
        self.linear = nn.Linear(config["hidden_size"], config["output_size"])


    def forward(self, simulations, questions, **kwargs):
        embeddings = self.embedding(questions)
        _, (hiddens, _) = self.lstm(embeddings)
        answers = self.linear(hiddens.squeeze(0))
        return answers


class VisualBaseline(nn.Module):
    """
    Does not use any kind of textual data.
    """

    SIMULATION_INPUT = None
    NUM_VIDEO_FRAMES = 0

    def __init__(self, config):
        super().__init__()
        self.convnet = eval(config["convnet"]["architecture"])(config["convnet"])
        config["mlp"]["input_size"] = self.convnet.out_features
        self.flatten = nn.Flatten()
        self.mlp = MLP(config["mlp"])
        self.linear = nn.Linear(
            in_features=config["mlp"]["hidden_size"],
            out_features=config["output_size"])
        self.config = config

    def forward(self, simulations, questions, **kwargs):
        y = self.convnet(simulations)
        y = self.flatten(y)
        y = self.mlp(y)
        return self.linear(y)


class FirstFrameVisualBaseline(VisualBaseline):
    """
    Input is only the first frame of the simulation video.
    """

    SIMULATION_INPUT = SimulationInput.FIRST_FRAME


class LastFrameVisualBaseline(VisualBaseline):
    SIMULATION_INPUT = SimulationInput.LAST_FRAME


class DoubleFramesVisualBaseline(VisualBaseline):
    SIMULATION_INPUT = SimulationInput.FIRST_AND_LAST_FRAMES

    def forward(self, simulations, questions, **kwargs):
        y = self.convnet(simulations)
        y = self.flatten(y)
        y = self.mlp(y)
        batch_size = y.shape[0] // 2
        first_frames, last_frames = y[:batch_size], y[:batch_size]
        return self.linear(first_frames + last_frames)


class SimpleBaseline(nn.Module):
    SIMULATION_INPUT = SimulationInput.NO_FRAMES
    NUM_VIDEO_FRAMES = 0

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["input_size"],
                                      config["embed_size"],
                                      padding_idx=0)
        self.lstm = nn.LSTM(config["embed_size"], config["hidden_size"])
        self.convnet = eval(config["convnet"]["architecture"])(config["convnet"])
        config["mlp"]["input_size"] = self.convnet.out_features
        if self.SIMULATION_INPUT == SimulationInput.FIRST_AND_LAST_FRAMES:
            config["mlp"]["input_size"] *= 2
        self.flatten = nn.Flatten()
        self.mlp = MLP(config["mlp"])
        in_features = config["mlp"]["hidden_size"] + config["hidden_size"]
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=config["output_size"])
        self.config = config

    def process_simulation(self, simulations, **kwargs):
        y = self.convnet(simulations)
        y = self.flatten(y)
        y = self.mlp(y)
        return y

    def process_question(self, questions, **kwargs):
        embeddings = self.embedding(questions)
        _, (hiddens, _) = self.lstm(embeddings)
        return hiddens.squeeze(0)

    def forward(self, simulations, questions, **kwargs):
        vis = self.process_simulation(simulations, **kwargs)
        txt = self.process_question(questions, **kwargs)
        return self.linear(torch.cat([vis, txt], dim=1))


class FirstFrameBaseline(SimpleBaseline):
    SIMULATION_INPUT = SimulationInput.FIRST_FRAME


class LastFrameBaseline(SimpleBaseline):
    SIMULATION_INPUT = SimulationInput.LAST_FRAME


class DoubleFramesBaseline(SimpleBaseline):
    SIMULATION_INPUT = SimulationInput.FIRST_AND_LAST_FRAMES

    def process_simulation(self, simulations, **kwargs):
        y = self.convnet(simulations)
        y = self.flatten(y)
        batch_size = y.size(0) // 2
        first_frames, last_frames = y[:batch_size], y[batch_size:]
        y = torch.cat([first_frames, last_frames], dim=1)
        y = self.mlp(y)
        return y


class VideoBaseline(SimpleBaseline):
    SIMULATION_INPUT = SimulationInput.VIDEO
    NUM_VIDEO_FRAMES = 10

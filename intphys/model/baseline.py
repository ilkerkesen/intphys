from copy import deepcopy

import torch
import torch.nn as nn

from ..submodule import *
from ..data import SimulationInput


__all__ = (
    "LSTMBaseline",
    "BERTBaseline",
    "DescriptionBaseline",
    "LSTMCNNBaselineFF",
    "LSTMCNNBaselineLF",
    "LSTMCNNBaseline2F",
    "LSTMCNNVideoBaseline",
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
        self.linear = nn.Linear(
            self.question_encoder.output_size, config["output_size"])
        self.dropout = nn.Dropout(p=config["dropout"])
        self.config = config

    def forward(self, simulations, questions, lengths, **kwargs):
        _, (hiddens, _) = self.question_encoder(questions, lengths)
        if self.question_encoder.lstm.bidirectional:
            hiddens = torch.cat([hiddens[0], hiddens[1]], dim=1)
        else:
            hiddens = hiddens.squeeze(0)
        answers = self.linear(self.dropout(hiddens))
        return answers


class BERTBaseline(nn.Module):
    SIMULATION_INPUT = SimulationInput.NO_FRAMES
    NUM_VIDEO_FRAMES = 0

    def __init__(self, config):
        super().__init__()
        transformer = config["question_encoder"].get("architecture")
        if transformer == "LSTMEncoder":
            transformer = "BERTEncoder"
        assert transformer in ("BERTEncoder", "LongformerEncoder")
        transformer = eval(transformer)
        self.question_encoder = transformer(config["question_encoder"])
        self.linear = nn.Linear(
            self.question_encoder.output_size,
            config["output_size"],
        )
        self.dropout = nn.Dropout(p=config["dropout"])
        self.config = config

    def forward(self, simulations, questions, lengths, **kwargs):
        embed = self.question_encoder(questions, lengths)
        hidden = embed.last_hidden_state[:, 0, :]
        hidden = self.dropout(hidden)
        output = self.linear(hidden)
        return output


class DescriptionBaseline(nn.Module):
    """
    Use the simulation description.
    """
    SIMULATION_INPUT = SimulationInput.NO_FRAMES
    NUM_VIDEO_FRAMES = 0

    def __init__(self, config):
        super().__init__()
        config["description_encoder"] = deepcopy(config["question_encoder"])
        config["question_encoder"]["vocab_size"] = config["input_size"]
        config["description_encoder"]["vocab_size"] = config["desc_vocab_size"]
        self.question_encoder = LSTMEncoder(config["question_encoder"])
        self.description_encoder = LSTMEncoder(config["description_encoder"])
        self.linear = nn.Linear(
            2*self.question_encoder.output_size, config["output_size"])
        self.dropout = nn.Dropout(p=config["dropout"])
        self.config = config

    def forward(self, simulations, questions, questions_l, **kwargs):
        descriptions = kwargs.get("descriptions")
        descriptions_l = kwargs.get("descriptions_l")
        assert descriptions is not None and descriptions_l is not None
        
        _, (q_hid, _) = self.question_encoder(questions, questions_l)
        if self.question_encoder.lstm.bidirectional:
            q_hid = torch.cat([q_hid[0], q_hid[1]], dim=1)
        else:
            q_hid = q_hid.squeeze(0)

        _, (d_hid, _) = self.description_encoder(descriptions, descriptions_l)
        if self.description_encoder.lstm.bidirectional:
            d_hid = torch.cat([d_hid[0], d_hid[1]], dim=1)
        else:
            d_hid = d_hid.squeeze(0)
        
        hid = torch.cat([d_hid, q_hid], dim=1)
        answers = self.linear(self.dropout(hid))
        return answers


class LSTMCNNBaseline(nn.Module):
    SIMULATION_INPUT = SimulationInput.NO_FRAMES
    NUM_VIDEO_FRAMES = 0

    def __init__(self, config):
        super().__init__()

        # input dependent params
        config["question_encoder"]["vocab_size"] = config["input_size"]
        config["frame_encoder"]["depth_size"] = config["depth_size"]
        config["frame_encoder"]["input_width"] = config["input_width"]
        config["frame_encoder"]["input_height"] = config["input_height"]
        self.config = config

        self.frame_encoder = self.create_submodule("frame_encoder")
        self.question_encoder = self.create_submodule("question_encoder")
        if self.SIMULATION_INPUT is not SimulationInput.VIDEO:
            self.adaptive_pool = nn.AdaptiveAvgPool2d(config["pool_size"])
        else:
            self.adaptive_pool = nn.AdaptiveAvgPool3d(
                (None, config["pool_size"], config["pool_size"])
            )
        visual_size = self.NUM_VIDEO_FRAMES * self.frame_encoder.out_channels * config["pool_size"]**2 
        if self.SIMULATION_INPUT is SimulationInput.VIDEO:
            visual_size *= self.frame_encoder.out_depth
        textual_size = self.question_encoder.output_size
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
        y = self.adaptive_pool(y)
        y = self.flatten(y)
        return y

    def process_question(self, questions, lengths, **kwargs):
        _, (hiddens, _) = self.question_encoder(questions, lengths)
        if self.question_encoder.lstm.bidirectional:
            hiddens = torch.cat([hiddens[0], hiddens[1]], dim=1)
        else:
            hiddens = hiddens.squeeze(0)
        return hiddens

    def forward(self, simulations, questions, lengths, **kwargs):
        vis = self.process_simulation(simulations, **kwargs)
        txt = self.process_question(questions, lengths, **kwargs)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_frame_encoder = self.frame_encoder
        self.last_frame_encoder = self.create_submodule("frame_encoder")

    def process_simulation(self, simulations, **kwargs):
        batch_size = simulations.size(0) // 2
        
        first_frames = self.first_frame_encoder(simulations[:batch_size])
        first_frames = self.flatten(self.adaptive_pool(first_frames))
        
        last_frames = self.last_frame_encoder(simulations[batch_size:])
        last_frames = self.flatten(self.adaptive_pool(last_frames))

        y = torch.cat([first_frames, last_frames], dim=1)
        return y


class LSTMCNNVideoBaseline(LSTMCNNBaseline):
    SIMULATION_INPUT = SimulationInput.VIDEO
    NUM_VIDEO_FRAMES = 1 # see parent class constructor

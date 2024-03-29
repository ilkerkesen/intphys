import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as _resnet18
from torchvision.models import resnet101 as _resnet101
from torchvision.models.video import r3d_18 as _r3d_18
from torchvision.models.video import r2plus1d_18 as _r2plus1d_18
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.vocab import GloVe
from transformers import AutoConfig, AutoModel, BertModel


GLOVE_DIM = 300


__all__ = (
    "MLP",
    "CNN2Dv1",
    "resnet18",
    "resnet101",
    "LSTMEncoder",
    "BERTEncoder",
    "LongformerEncoder",
    "r3d_18",
    "r2plus1d_18",
    "FiLMLayer",
    "FiLMBlock2D",
)


def get_glove_cache():
    path = osp.split(osp.realpath(__file__))[0]
    path = osp.abspath(osp.join(path, "..", ".vector_cache"))
    return path


def get_glove_vectors(vocab, cache=get_glove_cache()):
    glove = GloVe(name='840B', dim=300, cache=cache) 
    vectors = np.zeros((len(vocab), 300))
    count = 0
    for word, idx in vocab.w2i.items():
        vector = None
        if word == "<pad>" or "<PAD>":
            continue
        if word == "<unk>" or "<UNK>":
            word = "unk"
            vector = glove.vectors[glove.stoi[word]]
        if not (word in glove.stoi):
            count += 1
            vector = np.random.randn(GLOVE_DIM)
        if vector is None:
            vector = glove.vectors[glove.stoi[word]]
        vectors[idx, :] = vector
    return torch.tensor(vectors, dtype=torch.float)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential()
        func = config["activation"].lower()
        nonlinear = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh
        }[func]
        for i in range(config["num_layers"]):
            out_features = config["hidden_size"]
            in_features = config["input_size"] if i == 0 else out_features
            linear = nn.Linear(
                in_features=in_features,
                out_features=out_features
            )
            self.layers.add_module("dense{}".format(i), linear)
            self.layers.add_module("{}{}".format(func, i), nonlinear())

    def forward(self, x):
        return self.layers(x)


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.get("glove", False):
            config["embed_size"] = GLOVE_DIM
            vectors = get_glove_vectors(config["vocab"])
            self.embedding = nn.Embedding.from_pretrained(vectors)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=config["vocab_size"],
                embedding_dim=config["embed_size"],
                padding_idx=0)
        self.lstm = nn.LSTM(
            config["embed_size"],
            config["hidden_size"],
            bidirectional=config["bidirectional"],
            batch_first=True)
        self.config = config
        self.output_size = (1+config["bidirectional"]) * config["hidden_size"]

    def forward(self, x, x_l):
        embed = self.embedding(x)
        embed = pack_padded_sequence(
            embed, x_l, batch_first=True, enforce_sorted=False)
        return self.lstm(embed)


class BERTEncoder(nn.Module):
    MODEL = 'bert-base-uncased'
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_bert()
        self.bert = AutoModel.from_pretrained(self.MODEL)
        finetune = config.get('finetune', True)
        if not finetune:
            for p in self.bert.parameters():
                p.requires_grad = False
                
    def init_bert(self):
        if self.config.get('pretrained', True):
            self.bert = AutoModel.from_pretrained()
        else:
            bert_config = AutoConfig.from_pretrained(self.MODEL)
            self.bert = BertModel(bert_config)
            
    @property
    def is_pretrained(self):
        return self.config.get('pretrained', True)
    
    @property
    def output_size(self):
        return 768

    def forward(self, x, x_l):
        output = self.bert(input_ids=x, attention_mask=x_l)
        return output
    

class LongformerEncoder(BERTEncoder):
    MODEL = 'allenai/longformer-base-4096'
    

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
        w, h = self.config["input_width"], self.config["input_height"]
        for idx in range(self.config["num_layers"]):
            w, h = [self.calculate_output_size(x) for x in (w, h)]
            self.add_layer(idx=idx)
        num_channels = (2**idx) * self.config["num_channels"]
        self.out_features = w * h * num_channels
        self.out_channels = num_channels

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
        for par in net.parameters():
            par.requires_grad = False
    out_channels = 2**(5+config["num_layers"]-1)
    net.out_channels = out_channels
    net.config = config
    return net

    
def resnet101(config):
    net = _resnet101(pretrained=config["pretrained"], progress=True)
    layers = list(net.children())
    net = nn.Sequential(*layers[:4+config["num_layers"]-1])
    if config["pretrained"] and config["freeze"]:
        for par in net.parameters():
            par.requires_grad = False
    if config["num_layers"] > 1:
        net.out_channels = 64 * 2**(config["num_layers"])
    else:
        net.out_channels = 64
    net.config = config
    return net


def resnet3d(config, init=_r3d_18):
    net = init(pretrained=config["pretrained"], progress=True)
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
    net.out_depth = out_depth
    net.out_channels = out_channels
    net.config = config
    return net


def r3d_18(config):
    return resnet3d(config, init=_r3d_18)


def r2plus1d_18(config):
    return resnet3d(config, init=_r2plus1d_18)
    

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
# -*- coding: utf-8 -*-

import os
import os.path as osp

import torch
import torch.nn as nn
import pytorch_lightning as pl
from .model.baseline import *


class Experiment(pl.LightningModule):
    def __init__(self, config):
        super(Experiment, self).__init__()
        self.config = config
        self.model = eval(config["model"]["architecture"])(config["model"])
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(self.config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        questions, answers = batch
        output = self(questions)
        loss = self.criterion(output, answers)
        _, predictions = torch.max(output, 1)
        acc = (predictions == answers).float().mean()
        logs = {"train_loss": loss, "train_accuracy": acc}
        return {"loss": loss, "accuracy": acc}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        logs = {"train_loss": loss, "train_accuracy": acc}
        return {"train_loss": loss, "train_accuracy": acc,
                "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_index):
        return self.batch_accuracy(batch, batch_index)

    def validation_epoch_end(self, outputs):
        return self.calculate_accuracy(outputs, split="val")

    def test_step(self, batch, batch_index):
        return self.batch_accuracy(batch, batch_index)

    def test_epoch_end(self, outputs):
        return self.calculate_accuracy(outputs, split="test")

    def batch_accuracy(self, batch, batch_index):
        questions, answers = batch
        output = self(questions)
        _, predictions = torch.max(output, 1)
        correct = (answers == predictions).sum()
        num_instances = torch.tensor(answers.size(), device=self.device).float()
        loss = self.criterion(output, answers)
        return {"correct": correct,
                "num_instances": num_instances.squeeze(),
                "loss": loss}

    def calculate_accuracy(self, outputs, split):
        correct = torch.stack([x["correct"] for x in outputs]).sum()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        num_instances = sum([x["num_instances"] for x in outputs])
        acc = correct.float() / num_instances
        tensorboard_logs = {
            "{}_accuracy".format(split): acc,
            "{}_loss".format(split): loss}
        return {"{}_loss".format(split): loss,
                "{}_accuracy".format(split): acc,
                "log": tensorboard_logs,
                "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

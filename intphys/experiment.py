# -*- coding: utf-8 -*-

import os
import os.path as osp
from itertools import repeat

import torch
import torch.nn as nn
import pytorch_lightning as pl
from .model import *



__all__ = (
    'Experiment',
    'TSVExportCallback',
)


class Experiment(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = eval(config["model"]["architecture"])(config["model"])
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(self.config)
        self._generate_flag = False

    def forward(self, simulation, question, **kwargs):
        return self.model(simulation ,question, **kwargs)

    def training_step(self, batch, batch_index):
        (simulations, questions, additional), (answers,) = batch
        output = self(simulations, questions, **additional["kwargs"])
        loss = self.criterion(output, answers)
        _, predictions = torch.max(output, 1)
        acc = (predictions == answers).float().mean()
        logs = {"train_loss": loss, "train_accuracy": acc}
        return {"loss": loss, "accuracy": acc}

    def training_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        logs = {"train_loss": loss, "train_accuracy": acc}
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        #return {"train_loss": loss, "train_accuracy": acc,
        #        "log": logs, "progress_bar": logs}

    def validation_step(self, batch, batch_index):
        return self.batch_accuracy(batch, batch_index)

    def validation_epoch_end(self, outputs):
        split = self.val_dataloader().dataset.split
        split = "val" if split.startswith("val") else split
        return self.calculate_accuracy(outputs, split=split)

    def test_step(self, batch, batch_index):
        return self.batch_accuracy(batch, batch_index, testing=True)

    def test_epoch_end(self, outputs):
        split = self.test_dataloader().dataset.split
        return self.calculate_accuracy(outputs, split=split)

    def batch_accuracy(self, batch, batch_index, testing=False):
        (simulations, questions, additional), (answers,) = batch
        output = self(simulations, questions, **additional["kwargs"])
        _, predictions = torch.max(output, 1)
        correct = (answers == predictions).sum()
        num_instances = torch.tensor(answers.size(), device=self.device).float()
        loss = self.criterion(output, answers)

        if testing and self.generate_flag:
            self.write_generations(
                additional["video_indexes"],
                additional["question_indexes"],
                predictions)

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
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

    @property
    def generate_flag(self):
        return self._generate_flag

    @generate_flag.setter
    def generate_flag(self, flag):
        self._generate_flag = flag

    def write_generations(self, vids, qids, predictions):
        dataset = self.test_dataloader().dataset
        split, vocab = dataset.split, dataset.answer_vocab
        vocab = dataset.answer_vocab
        zipped = zip(vids, qids, repeat(split), vocab[predictions.tolist()])
        lines = ["\t".join([str(xi) for xi in x]) + "\n" for x in zipped]
        filepath = self.config["output"]
        with open(filepath, "a") as f:
            f.writelines(lines)


class TSVExportCallback(pl.Callback):
    def on_test_start(self, trainer, pl_module):
        if pl_module.generate_flag == False:
            return
        filepath = pl_module.config["output"]
        parent_dir = osp.dirname(filepath)
        osp.makedirs(parent_dir) if not osp.isdir(parent_dir) else None
        with open(filepath, "w") as f:
            f.write("video_index\tquestion_index\tsplit\tprediction\n")

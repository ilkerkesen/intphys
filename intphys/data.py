# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import re

import torch
import torch.utils.data as data
import numpy as np


UNK = "<UNK>"
PAD = "<PAD>"


PUNCTUATION_REGEX = re.compile(r"([a-zA-Z0-9]+)(\W)")
def tokenize_sentence(sentence):
    return PUNCTUATION_REGEX.sub(r"\g<1> \g<2>", sentence.lower()).split()


class Vocab(object):
    def __init__(self, instances, min_occur=0):
        self.min_occur = min_occur
        self.build_count_dict(instances)
        self.build_dicts()

    def build_count_dict(self, instances):
        count_dict = dict()
        for instance in instances:
            tokens = tokenize_sentence(str(instance))
            for token in tokens:
                count_dict[token] = count_dict.get(token, 0) + 1
        self.count_dict = count_dict

    def build_dicts(self):
        self.w2i, self.i2w = {PAD: 0}, {0: PAD}
        tokens = [word
                  for (word, count) in self.count_dict.items()
                  if count >= self.min_occur]
        tokens.sort()
        for (i,token) in enumerate(tokens):
            self.w2i[token] = i+1
            self.i2w[i+1] = token

        # add UNK token, just in case
        num_tokens = len(self.w2i)
        self.w2i[UNK] = num_tokens
        self.i2w[num_tokens] = UNK

    def __getitem__(self, x):
        x = (x,) if isinstance(x, int) else x
        x = tokenize_sentence(x) if isinstance(x, str) else x

        if all(isinstance(xi, str) for xi in x):
            x2i = self.w2i
        elif all(isinstance(xi, int) for xi in x):
            x2i = self.i2w
        else:
            raise Exception("Not all elements are same type for Vocab input.")
        x = [x2i.get(xi, x2i.get(UNK)) for xi in x]
        return x


class IntuitivePhysicsDataset(data.Dataset):
    def __init__(self, datadir, split="train"):
        self.read_jsonfile(datadir, split)
        self.build_vocabs()

    def read_jsonfile(self, datadir, split):
        with open(osp.join(datadir, "dataset.json")) as f:
            json_data = json.load(f)
        simulations = list(filter(lambda x: x["split"] == split, json_data))
        self.questions = []
        for sim in simulations:
            self.questions.extend(sim["questions"]["questions"])

    def build_vocabs(self):
        self.question_vocab = Vocab([x['question'] for x in self.questions])
        self.answer_vocab = Vocab([x['answer'] for x in self.questions])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = self.questions[idx]
        question = self.question_vocab[item["question"]]
        answer = self.answer_vocab[tokenize_sentence(str(item["answer"]))]
        return (torch.tensor(question), torch.tensor(answer))


def collate_fn(unsorted_batch):
    batch = sorted(unsorted_batch, key=lambda x: len(x[0]), reverse=True)
    batchsize, longest = len(batch), len(batch[0][0])
    questions = torch.zeros((batchsize, longest), dtype=torch.long)
    for (i, (question, _)) in enumerate(batch):
        questions[i, -len(question):] = question
    answers = torch.cat([answer for (_, answer) in batch]).unsqueeze(1)
    return (questions, answers)

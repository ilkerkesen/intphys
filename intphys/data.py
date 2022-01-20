# -*- coding: utf-8 -*-

import os
import os.path as osp
import json
import re
from enum import IntEnum
from copy import deepcopy

import torch
import torch.utils.data as data
from torchvision.io import read_video, read_video_timestamps
import numpy as np
import cv2
from tqdm import tqdm
from torchtext.data import get_tokenizer
from transformers import AutoTokenizer
from intphys.submodule import BERTEncoder, LongformerEncoder


UNK = "<UNK>"
PAD = "<PAD>"


__all__ = (
    'SimulationInput',
    'CRAFT',
    'CLEVRER',
    'train_collate_fn',
    'inference_collate_fn',
)


def rearrange_dimensions(frames):
    # make channel dimension 1st dimension
    new_frames = frames.unsqueeze(0).transpose(0, -1).squeeze(-1)

    # remove depth in case of single frame
    C, D, H, W = new_frames.shape
    new_frames = new_frames.squeeze() if D == 1 else new_frames

    return new_frames


class SimulationInput(IntEnum):
    NO_FRAMES = 0
    FIRST_FRAME = 1
    LAST_FRAME = 2
    FIRST_AND_LAST_FRAMES = 3
    VIDEO = 4
    DESCRIPTION = 5

    def from_string(obj):
        if isinstance(obj, int) or isinstance(obj, SimulationInput):
            return SimulationInput(obj)
        obj = "SimulationInput.{}".format(obj)
        symbol = eval(obj)
        return SimulationInput(symbol)


class Vocab(object):
    def __init__(self, instances, min_occur=0, ngram=1):
        self.min_occur = min_occur
        self.ngram = ngram
        self.tokenizer = get_tokenizer("basic_english")
        self.build_count_dict(instances)
        self.build_dicts()

    def build_count_dict(self, instances):
        count_dict = dict()
        for instance in instances:
            tokens = self.tokenizer(str(instance))
            for i in range(len(tokens)+1-self.ngram):
                ngram = tokens[i:i+self.ngram]
                ngram = " ".join(ngram)
                count_dict[ngram] = count_dict.get(ngram, 0) + 1
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
        x = x.tolist() if isinstance(x, torch.Tensor) else x
        if isinstance(x, str):
            x, n = self.tokenizer(x), self.ngram
            x = [" ".join(x[i:i+n]) for i in range(len(x)+1-n)]

        if all(isinstance(xi, str) for xi in x):
            x2i = self.w2i
        elif all(isinstance(xi, int) for xi in x):
            x2i = self.i2w
        else:
            raise Exception("Not all elements are same type for Vocab input.")
        x = [x2i.get(xi, x2i.get(UNK)) for xi in x]
        return x

    def __len__(self):
        return len(self.w2i)


class BaseDataset(data.Dataset):
    NUM_SECONDS = 0
    HEIGHT = 0
    WIDTH = 0

    def __init__(self, path, split="train", fps=0, cached=False,
                 transform=None, split_info="random", ngram=1,
                 description_mode="simplest", *args, **kwargs):
        super().__init__()
        self.datadir = osp.abspath(osp.expanduser(path))
        self.split = split
        self.split_info = split_info
        self.fps = fps
        self.normalizer = None
        self.sim_input = None
        self.cached = cached
        self.transform = transform
        self.use_bert = False
        self.ngram = ngram

        assert description_mode in ("full", "simple", "simplest")        
        self.description_mode = description_mode

        self.read_jsonfile()
        self.build_vocabs()
        self.build_split()
    
    def adapt2model(self, model):
        self.sim_input = model.SIMULATION_INPUT
        try:
            self.normalizer = model.frame_encoder.normalizer
        except AttributeError:
            pass

        if isinstance(model.question_encoder, BERTEncoder):
            self.use_bert = True
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif isinstance(model.question_encoder, LongformerEncoder):
            self.use_bert = True
            self.bert_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        
        has_description_encoder = hasattr(model, 'description_encoder')
        insert_descriptions = model.config.get('use_descriptions', False)
        self.use_descriptions = has_description_encoder or insert_descriptions

    def read_jsonfile(self):
        raise NotImplementedError

    def build_vocabs(self):
        raise NotImplementedError

    def build_split(self):
        raise NotImplementedError

    def read_simulation(self, path):
        sim_input = str(self.sim_input).split(".")[1]
        sim_func = eval("self.read_{}".format(sim_input.lower()))
        return sim_func(path)

    def read_frame(self, path, frame="first"):
        image = cv2.imread(self.get_frame_path(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image).permute(2, 0, 1)
        return self.postprocess_simulation(image)

    def read_first_frame(self, path):
        return self.read_frame(path)

    def read_last_frame(self, path):
        return self.read_frame(path, frame="last")

    def read_first_and_last_frames(self, path):
        first_frame = self.read_first_frame(path)
        last_frame = self.read_last_frame(path)
        return (first_frame, last_frame)

    def read_video(self, path):
        # filename = item["video_file_path"]
        video = read_video(self.get_video_path(path), pts_unit="sec")[0]
        video = rearrange_dimensions(video)
        return self.postprocess_simulation(video)

    def read_no_frames(self, item):
        return (torch.zeros(1),)

    def postprocess_simulation(self, simulation):
        processed = simulation / 255.0

        # normalization
        if self.normalizer is not None:
            processed = self.normalizer(processed)

        # transformation after normalization
        if self.transform is not None and processed.dim() == 4:
            frames = [processed[:, i] for i in range(processed.shape[1])]
            frames = [self.transform(frame).unsqueeze(1) for frame in frames]
            processed = torch.cat(frames, dim=1)
        elif self.transform is not None and processed.dim() != 3:
            processed = self.transform(processed)

        # make it appropriate for minibatching
        processed = processed.unsqueeze(0)
        return processed

    def get_frame_path(self, filepath, frame="first"):
        raise NotImplementedError

    def get_video_path(self, filepath):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class CRAFT(BaseDataset):
    NUM_SECONDS = 10
    HEIGHT = 256
    WIDTH = 256
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_descriptions()
        self.use_descriptions = False

    def read_jsonfile(self):
        with open(osp.join(self.datadir, "dataset_minimal.json")) as f:
            self.json_data = json.load(f)

    def build_vocabs(self):
        self.question_vocab = Vocab(
            [x['question'] for x in self.json_data],
            ngram=self.ngram)
        self.answer_vocab = Vocab([x['answer'] for x in self.json_data])
        self.choice_vocab = Vocab([])

    def build_split(self, ):
        filepath = osp.join(self.datadir, f"split_info_{self.split_info}.json")
        with open(filepath) as f:
            split_data = json.load(f)
        split_dict = {}
        for (split, instances) in split_data.items():
            for x in instances:
                split_dict[x["video_index"], x['question_index']] = split
        for x in self.json_data:
            x["split"] = split_dict[x["video_index"], x["question_index"]]

        questions = deepcopy(self.json_data)
        if self.split != "all":
            questions = [x for x in self.json_data if x["split"] == self.split]
        self.questions = questions

    def build_cache(self):
        self.cache = dict()
        questions = self.questions
        items = {(q["video_index"], q["video_file_path"])
                 for q in questions}
        items = [{"video_index": x[0], "video_file_path": x[1]}
                 for x in items]
        for item in tqdm(items):
            if item["video_index"] in self.cache.keys(): continue
            self.cache[item['video_index']] = self.read_simulation(item["video_file_path"])
            
    def load_descriptions(self):
        filepath = osp.join(
            self.datadir,
            "descriptions",
            f"{self.description_mode}.json",
        )
        with open(filepath) as f:
            self.descriptions = json.load(f)

        self.description_vocab = Vocab(
            [v for v in self.descriptions.values()],
            ngram=self.ngram)
        
    def get_frame_path(self, path, frame="first"):
        path = path.replace("videos", frame + "_frames").replace("mpg", "png")
        path = osp.abspath(osp.join(self.datadir, path))
        return path

    def get_video_path(self, path):
        if self.fps > 0:
            path = path.replace("videos", f"downsampled/{self.fps}-fps")
            path = path.replace(".mpg", ".mp4")
        path = osp.abspath(osp.join(self.datadir, path))
        return path
    
    def truncate(self, input_dict):
        if self.bert_tokenizer.name_or_path != 'bert-base-uncased':
            return input_dict
        
        max_len = 512
        if len(input_dict['input_ids']) <= max_len:
            return input_dict
        
        input_ids = input_dict['input_ids']
        input_ids = input_ids[:1] + input_ids[-max_len+1:]
        attention_mask = input_dict['attention_mask']
        attention_mask = attention_mask[:1] + attention_mask[-max_len+1:]
        input_dict['input_ids'] = input_ids
        input_dict['attention_mask'] = attention_mask
        return input_dict
    
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = self.questions[idx]
        
        # read simulation
        if self.cached and item["video_index"] in self.cache.keys():
            simulation = self.cache[item["video_index"]]
        elif not self.cached:
            simulation = self.read_simulation(item["video_file_path"])
        else:
            print("read simulation op: split={}, video_index={}".format(
                self.split, item["video_index"]))
        if isinstance(simulation, torch.Tensor):
            simulation = (simulation,)
        
        # read question / answer
        if self.use_bert:
            question = item["question"]
            if self.use_descriptions:
                video_index = f"{item['video_index']:06d}"
                description = self.descriptions[video_index]
                question = description + ' ' + question
            input_dict = self.bert_tokenizer(question)
            input_dict = self.truncate(input_dict)
            question = input_dict["input_ids"]
            length = input_dict["attention_mask"]
        else:
            question = self.question_vocab[item["question"]]
            length = len(question)
        answer = self.answer_vocab[self.answer_vocab.tokenizer(str(item["answer"]))]
        
        # read descriptions
        description = description_l = None
        if self.use_descriptions and not self.use_bert:
            video_index = f"{item['video_index']:06d}"
            description = self.descriptions[video_index]
            description = self.description_vocab[description]
            description_l = len(description)
            description = torch.tensor(description)
        
        item_dict = {
            "simulation": simulation,
            "question": torch.tensor(question),
            "question_length": length,
            "answer": torch.tensor(answer),
            "answer_type": item["answer_type"].lower(),
            "template": item["question_type"].lower(),
            "video_index": item["video_index"],
            "question_index": item["question_index"],
            "description": description,
            "description_length": description_l,
        }
        return item_dict


class CLEVRER(BaseDataset):
    NUM_SECONDS = 5
    WIDTH = 480
    HEIGHT = 320

    def read_jsonfile(self):
        path = osp.join(self.datadir, self.split + ".json")
        with open(path) as f:
            self.json_data = json.load(f)
    
    def build_vocabs(self):
        vocab_file = osp.join(self.datadir, "vocab.pt")
        if osp.isfile(vocab_file):
            vocabs = torch.load(vocab_file)
            self.question_vocab = vocabs["question_vocab"]
            self.answer_vocab = vocabs["answer_vocab"]
            return
        
        with open(osp.join(self.datadir, "train.json")) as f:
            json_data = json.load(f)
        
        questions, choices, answers = [], [], []
        for simulation in json_data:
            for question in simulation["questions"]:
                questions.append(question["question"])
                if "answer" in question.keys():
                    answers.append(question["answer"])
                    continue

                for choice in question["choices"]:
                    choices.append(choice["choice"])
                    answers.append(choice["answer"])
        
        self.question_vocab = Vocab(questions + choices)
        self.answer_vocab = Vocab(answers)

        torch.save({
            "question_vocab": self.question_vocab,
            "answer_vocab": self.answer_vocab,
        }, vocab_file)

    def build_split(self):
        self.questions = []
        for simulation in self.json_data:
            base_dict = {
                "video_file_path": simulation["video_file_path"],
                "scene_index": simulation["scene_index"],
            }
            for question in simulation['questions']:
                question_dict = {k:v for (k,v) in question.items() if k != "choices"}
                if "answer" in question.keys():
                    self.questions.append({**base_dict, **question_dict})
                    continue
                
                for choice_dict in question["choices"]:
                    self.questions.append({
                        **base_dict,
                        **question_dict,
                        **choice_dict
                    })

    def get_video_fullpath(self, path):
        idx = int(re.match(r"video_(\d+)\.mp4", path).group(1))
        idx = (idx // 1000) * 1000
        path = osp.join(f"videos/video_{idx:05d}-{idx+1000:05d}", path)
        return osp.join(self.datadir, path)

    def get_frame_path(self, path, frame="first"):
        return path.replace("videos", f"{frame}_frames").replace("mp4", "png")

    def get_video_path(self, path):
        if self.fps > 0:
            path = path.replace("videos", f"downsampled/{self.fps}fps")
        return path

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        item = self.questions[idx]
        merged = item["question"] + " " + item.get("choice", "")
        question = self.question_vocab[merged]
        answer = self.answer_vocab[item["answer"]]
        video_path = self.get_video_fullpath(item["video_file_path"])
        simulation = self.read_simulation(video_path)
        if isinstance(simulation, torch.Tensor): simulation = (simulation, )

        item_dict = {
            "simulation": simulation,
            "question": torch.tensor(question),
            "template": item["question_type"],
            "answer": torch.tensor(answer),
            "video_index": item["scene_index"],
            "question_index": item["question_id"],
        }

        return item_dict


def make_sentence_batch(batch, batch_first=True, pad_right=True):
    batchsize = len(batch)
    longest = max([len(x) for x in batch])
    sentences = torch.zeros((batchsize, longest), dtype=torch.long)
    for (i, sentence) in enumerate(batch):
        if not pad_right:
            sentences[i, -len(sentence):] = sentence
        else:
            sentences[i, :len(sentence)] = sentence
    if not batch_first:
        sentences = sentences.transpose(0, 1)
    return sentences


def base_collate_fn(batch):
    # question batching
    if isinstance(batch[0]["question_length"], int):
        questions = make_sentence_batch([x["question"] for x in batch])
        lengths = [x["question_length"] for x in batch]
    else:
        lengths = torch.zeros(len(batch), len(batch[0]["question_length"]))
        for i, bi in enumerate(batch):
            mask = bi["question_length"]
            lengths[i, :len(mask)] = torch.tensor(mask)
        questions = make_sentence_batch(
            [x["question"] for x in batch], pad_right=True)

    # answer batching
    answers = torch.cat([instance["answer"] for instance in batch])

    # simulation batching
    num_simulations = len(batch[0]["simulation"])
    helper = lambda i: torch.cat([x["simulation"][i] for x in batch], dim=0)
    simulations = torch.cat([helper(i) for i in range(num_simulations)], dim=0)

    return (simulations, questions, lengths, answers)


def train_collate_fn(unsorted_batch):
    unsorted_batch = [x for x in unsorted_batch if x is not None]
    batch = sorted(unsorted_batch,
                   # key=lambda x: len(x["question"], x["choice"]),
                   key=lambda x: len(x["question"]),
                   reverse=True)
    simulations, questions, lengths, answers = base_collate_fn(batch)
    
    # description batching
    descriptions = descriptions_l = None
    if batch[0].get("description") is not None:
        descriptions = make_sentence_batch([x["description"] for x in batch])
        descriptions_l = [x["description_length"] for x in batch]    
    
    additional = {"kwargs": {
        "descriptions": descriptions,
        "descriptions_l": descriptions_l,
    }}
    inputs, outputs = (simulations, questions, lengths, additional), (answers,)
    return (inputs, outputs)


def inference_collate_fn(unsorted_batch):
    unsorted_batch = [x for x in unsorted_batch if x is not None]
    batch = sorted(unsorted_batch,
                   key=lambda x: len(x["question"]),
                   reverse=True)
    simulations, questions, lengths, answers = base_collate_fn(batch)
    
    # description batching
    descriptions = descriptions_l = None
    if batch[0].get("description") is not None:
        descriptions = make_sentence_batch([x["description"] for x in batch])
        descriptions_l = [x["description_length"] for x in batch]    
    
    additional = {
        "video_indexes": [x["video_index"] for x in batch],
        "question_indexes": [x["question_index"] for x in batch],
        "kwargs": {
            "descriptions": descriptions,
            "descriptions_l": descriptions_l,
        },
    }
    inputs, outputs = (simulations, questions, lengths, additional), (answers,)
    return (inputs, outputs)

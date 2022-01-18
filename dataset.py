import json
import os
import copy
from torch.utils.data import Dataset
from dictionary import Dictionary
import torch
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
import random

class Seq2SeqDataset(Dataset):
    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", args=None):
        self.data_path = data_path
        self.src_file = os.path.join(data_path, "in_"+args.trainset+".txt")
        if args.loop:
            self.tgt_file = os.path.join(data_path, "out_"+args.trainset+"_loop.txt")
        else:
            self.tgt_file = os.path.join(data_path, "out_"+args.trainset+".txt")
            
        with open(self.src_file) as fsrc, open(self.tgt_file) as ftgt:
            self.src_lines = fsrc.readlines()
            self.tgt_lines = ftgt.readlines()

        assert len(self.src_lines) == len(self.tgt_lines)
        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)
        self.smart_filter = args.smart_filter
        self.args = args
    
    def __len__(self):
        return len(self.src_lines)

    def _init_vocab(self):
        self.dictionary.add_symbol('LOOP')
        N = 0
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                N += 1
        with open(self.data_path+'relation2id.txt') as fin:
            for line in fin:
                r, rid = line.strip().split('\t')
                rev_rid = int(rid) + N
                self.dictionary.add_symbol('R'+rid)
                self.dictionary.add_symbol('R'+str(rev_rid))
        with open(self.data_path+'entity2id.txt') as fin:
            for line in fin:
                e, eid = line.strip().split('\t')
                self.dictionary.add_symbol(eid)
        self.dictionary.save(self.vocab_file)

    def __getitem__(self, index):
        src_line = self.src_lines[index].strip().split(' ')
        tgt_line = self.tgt_lines[index].strip().split(' ')
        source_id = self.dictionary.encode_line(src_line)
        target_id = self.dictionary.encode_line(tgt_line)
        l = len(target_id)
        mask = torch.ones_like(target_id)
        for i in range(0, l-2):
            if i % 2 == 0: # do not mask relation
                continue
            if random.random() < self.args.prob: # randomly replace with prob
                target_id[i] = random.randint(0, self.len_vocab - 1)
                mask[i] = 0
        return {
            "id": index,
            "tgt_length": len(target_id),
            "source": source_id,
            "target": target_id,
            "mask": mask,
        }

    def collate_fn(self, samples):
        lens = [sample["tgt_length"] for sample in samples]
        max_len = max(lens)
        bsz = len(lens)
        source = torch.LongTensor(bsz, 3)
        prev_outputs = torch.LongTensor(bsz, max_len)
        mask = torch.zeros(bsz, max_len)

        source[:, 0].fill_(self.dictionary.bos())
        prev_outputs.fill_(self.dictionary.pad())
        prev_outputs[:, 0].fill_(self.dictionary.bos())
        target = copy.deepcopy(prev_outputs)

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            source_ids = sample["source"]
            target_ids = sample["target"]

            source[idx, 1:] = source_ids[: -1]
            prev_outputs[idx, 1:sample["tgt_length"]] = target_ids[: -1]
            target[idx, 0: sample["tgt_length"]] = target_ids
            mask[idx, 0: sample["tgt_length"]] = sample["mask"]

        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "lengths": torch.LongTensor(lens).to(self.device),
            "source": source.to(self.device),
            "prev_outputs": prev_outputs.to(self.device),
            "target": target.to(self.device),
            "mask": mask.to(self.device),
        }

    def get_next_valid(self):
        train_valid = dict()
        eval_valid = dict()
        vocab_size = len(self.dictionary)
        eos = self.dictionary.eos()
        with open(self.data_path+'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in train_valid:
                    if self.smart_filter:
                        train_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[e] = [eos, ]
                if er not in train_valid:
                    if self.smart_filter:
                        train_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        train_valid[er] = []
                if self.smart_filter:
                    train_valid[e][rid] = 0
                    train_valid[e][eos] = 0
                    train_valid[er][tid] = 0
                else:
                    train_valid[e].append(rid)
                    train_valid[er].append(tid)
        with open(self.data_path+'train_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                e = hid
                er = vocab_size * rid + hid
                if e not in eval_valid:
                    if self.smart_filter:
                        eval_valid[e] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[e] = [eos, ]
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[e][rid] = 0
                    eval_valid[e][eos] = 0
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[e].append(rid)
                    eval_valid[er].append(tid)
        with open(self.data_path+'valid_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)
        with open(self.data_path+'test_triples_rev.txt', 'r') as f:
            for line in tqdm(f):
                h, r, t = line.strip().split('\t')
                hid = self.dictionary.indices[h]
                rid = self.dictionary.indices[r]
                tid = self.dictionary.indices[t]
                er = vocab_size * rid + hid
                if er not in eval_valid:
                    if self.smart_filter:
                        eval_valid[er] = -30 * torch.ones([vocab_size])
                    else:
                        eval_valid[er] = []
                if self.smart_filter:
                    eval_valid[er][tid] = 0
                else:
                    eval_valid[er].append(tid)
        return train_valid, eval_valid
                
class TestDataset(Dataset):
    def __init__(self, data_path="FB15K237/", vocab_file="FB15K237/vocab.txt", device="cpu", src_file=None):

        if src_file:
            self.src_file = os.path.join(data_path, src_file)
        else:
            self.src_file = os.path.join(data_path, "valid_triples.txt")
            
        with open(self.src_file) as f:
            self.src_lines = f.readlines()

        self.vocab_file = vocab_file
        self.device = device
    
        try:
            self.dictionary = Dictionary.load(vocab_file)
        except FileNotFoundError:
            self.dictionary = Dictionary()
            self._init_vocab()
        self.padding_idx = self.dictionary.pad()
        self.len_vocab = len(self.dictionary)

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, index):
        
        src_line = self.src_lines[index].strip().split('\t')
        source_id = self.dictionary.encode_line(src_line[:2])
        target_id = self.dictionary.encode_line(src_line[2:])
        return {
            "id": index,
            "source": source_id,
            "target": target_id,
        }

    def collate_fn(self, samples):
        bsz = len(samples)
        source = torch.LongTensor(bsz, 3)
        target = torch.LongTensor(bsz, 1)

        source[:, 0].fill_(self.dictionary.bos())

        ids =  []
        for idx, sample in enumerate(samples):
            ids.append(sample["id"])
            source_ids = sample["source"]
            target_ids = sample["target"]
            source[idx, 1:] = source_ids[: -1]
            target[idx, 0] = target_ids[: -1]
        
        return {
            "ids": torch.LongTensor(ids).to(self.device),
            "source": source.to(self.device),
            "target": target.to(self.device)
        }
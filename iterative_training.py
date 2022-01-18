import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import Seq2SeqDataset, TestDataset
from model import TransformerModel
import argparse
import numpy as np
import os
from tqdm import tqdm
import logging
import transformers
import random
import networkx as nx

class Iter_trainer(object):
    def __init__(self, dataset, batch_size, beam_size, out_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.out_size = out_size
        self.rel_num = 0
    
    @staticmethod
    def read_triple(file_path, entity2id, relation2id):
        '''
        Read triples and map them into ids.
        '''
        triples = []
        with open(file_path) as fin:
            for line in fin:
                h, r, t = line.strip().split('\t')
                triples.append((entity2id[h], relation2id[r], entity2id[t]))
        return triples

    @staticmethod
    def find_path(connected, start, end, max_len, extra_start):
        paths = []
        if (start not in connected) or (end not in connected):
            return [[]]
        # one-hop
        path1 = []
        if end in connected[start]:
            for label in connected[start][end]:
                path1.append([label, end])
        paths.append(path1)
        if max_len == 1:
            return paths
        # two-hop
        path2 = []
        for mid in connected[start]:
            if mid == end or mid == start or mid == extra_start:
                continue
            if end not in connected[mid]:
                continue
            labels1 = connected[start][mid]
            labels2 = connected[mid][end]
            for label2 in labels2:
                for label1 in labels1:
                    path2.append([label1, mid, label2, end])
        paths.append(path2)
        if max_len == 2:
            return paths
        # three-hop
        path3 = []
        for mid1 in connected[start]:
            if mid1 == end or mid1 == start:
                continue
            for mid2 in connected[mid1]:
                if mid2 == end or mid2 == start or mid2 == mid1:
                    continue
                if end not in connected[mid2]:
                    continue
                labels1 = connected[start][mid1]
                labels2 = connected[mid1][mid2]
                labels3 = connected[mid2][end]
                for label3 in labels3:
                    for label2 in labels2:
                        for label1 in labels1:
                            path3.append([label1, mid1, label2, mid2, label3, end])
        paths.append(path3)
        if max_len == 3:
            return paths

    @staticmethod
    def search_path(connected, start, end, max_len, extra_start):
        candidate_paths = [[start, ]]
        paths = [[] for i in range(max_len+1)]
        while len(candidate_paths):
            path = candidate_paths.pop(0)
            pre = path[-1]
            l = int(len(path)/2)
            if pre == end and l > 0: # find path
                paths[l].append(path[1:])
                continue
            if l == max_len - 1:
                if pre in connected and end in connected[pre]:
                    for edge in connected[pre][end]:
                        paths[max_len].append(path[1:] + [edge, end])
            else:
                if pre in connected:
                    for suc in connected[pre].keys():
                        if (suc not in path[::2]) and (suc != extra_start):
                            for edge in connected[pre][suc]:
                                candidate_paths.append(path + [edge, suc])
        return paths

    def create_graph(self):
        lines = []
        entity2id = {}
        relation2id = {}
        with open(os.path.join(self.dataset, 'entity2id.txt')) as fin:
            for line in fin:
                e, eid = line.strip().split('\t')
                entity2id[e] = int(eid)

        with open(os.path.join(self.dataset, 'relation2id.txt')) as fin:
            for line in fin:
                r, rid = line.strip().split('\t')
                relation2id[r] = int(rid)
        
        self.rel_num = len(relation2id)
        G = nx.DiGraph()
        train_triples = self.read_triple(os.path.join(self.dataset, 'train.txt'), entity2id, relation2id)
        valid_triples = self.read_triple(os.path.join(self.dataset, 'valid.txt'), entity2id, relation2id)
        test_triples = self.read_triple(os.path.join(self.dataset, 'test.txt'), entity2id, relation2id)
        all_true_triples = train_triples #+ valid_triples + test_triples
        all_reverse_triples = []
        connected = dict() # {start: {end1: [edge11, edge12, ...], end2: [edge21, edge22, ...], ...}, ...}
        for triple in all_true_triples:
            all_reverse_triples.append((triple[2], triple[1]+self.rel_num, triple[0]))
        all_triples = all_true_triples + all_reverse_triples
        for triple in all_triples:
            start = triple[0]
            end = triple[2]
            edge = triple[1]
            if start not in connected:
                connected[start] = dict()
            if end not in connected[start]:
                connected[start][end] = []
            connected[start][end].append(edge)
        return connected
    
    def inv(self, edge):
        if edge >= self.rel_num:
            return (edge - self.rel_num)
        else:
            return (edge + self.rel_num)

    def get_iter(self, model, it):
        random.seed(12345)
        in_line = list()
        out_line = list()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_set = TestDataset(data_path=self.dataset+"/", vocab_file=self.dataset+"/vocab.txt", device=device, src_file="train_triples_rev.txt")
        train_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
        beam_size = self.beam_size
        model.eval()
        vocab_size = len(model.dictionary)
        eos = model.dictionary.eos()
        bos = model.dictionary.bos()
        # one-hop
        max_len = 2 * it + 1
        nsample = 6
        connected = self.create_graph()
        rev_dict = dict()
        for name in model.dictionary.indices.keys():
            index = model.dictionary.indices[name]
            rev_dict[index] = name

        with tqdm(train_loader, desc="iterating") as pbar:
            for samples in pbar:
                batch_size = samples["source"].size(0)
                candidate_path = [list() for i in range(batch_size)]
                candidate_score = [list() for i in range(batch_size)]
                source = samples["source"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
                prefix = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
                lprob = torch.zeros([batch_size, beam_size]).to(device)
                clen = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
                # first token: choose beam_size from only vocab_size, initiate prefix
                tmp_source = samples["source"]
                tmp_prefix = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
                tmp_prefix[:, 0].fill_(model.dictionary.bos())
                logits = model.logits(tmp_source, tmp_prefix).squeeze()
                logits = F.log_softmax(logits, dim=-1)
                
                argsort = torch.argsort(logits, dim=-1, descending=True)[:, :beam_size]
                prefix[:, :, 1] = argsort[:, :]
                lprob += torch.gather(input=logits, dim=-1, index=argsort)
                clen += 1
                for l in range(2, max_len):
                    tmp_prefix = prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                    tmp_lprob = lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)      
                    tmp_clen = clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                    bb = batch_size * beam_size
                    all_logits = model.logits(source.view(bb, -1), prefix.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                    logits = torch.gather(input=all_logits, dim=2, index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                    logits = F.log_softmax(logits, dim=-1)

                    argsort = torch.argsort(logits, dim=-1, descending=True)[:, :, :beam_size]
                    tmp_clen = tmp_clen + 1
                    tmp_prefix = tmp_prefix.scatter_(dim=-1, index=tmp_clen.unsqueeze(-1), src=argsort.unsqueeze(-1))
                    tmp_lprob += torch.gather(input=logits, dim=-1, index=argsort)
                    tmp_prefix, tmp_lprob, tmp_clen = tmp_prefix.view(batch_size, -1, max_len), tmp_lprob.view(batch_size, -1), tmp_clen.view(batch_size, -1)
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :beam_size]
                    prefix = torch.gather(input=tmp_prefix, dim=1, index=argsort.unsqueeze(-1).repeat(1, 1, max_len))
                    lprob = torch.gather(input=tmp_lprob, dim=1, index=argsort)
                    clen = torch.gather(input=tmp_clen, dim=1, index=argsort)
                    # filter out next token after <end>, add to candidates                    
                    for i in range(batch_size):
                        for j in range(beam_size):
                            if prefix[i][j][l].item() == eos:
                                prob = torch.exp(lprob[i][j]).item()
                                lprob[i][j] -= 10000
                                candidate_path[i].append([prefix[i][j][t].item() for t in range(1, l)])
                                candidate_score[i].append(prob)
                    # no <end> but reach max_len
                    if l == max_len-1:
                        for i in range(batch_size):
                            for j in range(self.out_size):
                                candidate = prefix[i][j][l].item()
                                prob = torch.exp(lprob[i][j]).item()
                                candidate_path[i].append([prefix[i][j][t].item() for t in range(1, max_len)])
                                candidate_score[i].append(prob)
                source = samples["source"].cpu()
                target = samples["target"].cpu()
                # output path
                for i in range(batch_size):
                    hid = source[i][1].item()
                    rid = source[i][2].item()
                    tid = target[i][0].item()
                    start = int(rev_dict[hid])
                    edge = int(rev_dict[rid][1:])
                    end = int(rev_dict[tid])
                    connected[start][end].remove(edge)
                    candidate_score[i] = np.array(candidate_score[i])
                    candidate_score[i] /= np.sum(candidate_score[i])
                    paths = None
                    total_paths = None
                    rev = False
                    for j in range(nsample):
                        # sample a candidate path from candidate_path according to candidate_score
                        prob = random.random()
                        candidate = -1
                        while prob > 0:
                            candidate += 1
                            prob -= candidate_score[i][candidate]
                        path = candidate_path[i][candidate]
                        mid = rev_dict[path[-1]]
                        if mid.isdigit():
                            mid = int(mid)
                        # judge mid
                        if mid == end:
                            length = int(len(path) / 2)
                            out_path = ""
                            for t in range(length):
                                if t > 0:
                                    out_path += ' '
                                out_path += (rev_dict[path[2*t]]+' '+rev_dict[path[2*t+1]])
                            out_path += '\n'
                            out_line.append(out_path)
                            in_line.append(str(start)+' R'+str(edge)+'\n')
                        else:
                            out_paths = self.find_path(connected, mid, end, 3-int(len(path)/2), start)
                            count = 0
                            for out_path in out_paths:
                                count += len(out_path)
                            if count > 0: # can find one after path
                                N = len(out_paths)
                                extra_path = None
                                while not extra_path:
                                    choice = random.randint(0, N-1)
                                    M = len(out_paths[choice])
                                    if M > 0:
                                        choice1 = random.randint(0, M-1)
                                        extra_path = out_paths[choice][choice1]
                                length1 = int(len(path)/2)
                                length2 = int(len(extra_path)/2)
                                out_path = ""
                                for t in range(length1):
                                    if t > 0:
                                        out_path += ' '
                                    out_path += (rev_dict[path[2*t]]+' '+rev_dict[path[2*t+1]])
                                for t in range(length2):
                                    out_path += (' R'+str(extra_path[2*t])+' '+str(extra_path[2*t+1]))
                                out_path += '\n'
                                out_line.append(out_path)
                                in_line.append(str(start)+' R'+str(edge)+'\n')
                            else: # cannot find after path, then generate from scratch
                                if not total_paths:
                                    total_paths = self.find_path(connected, start, end, 3, -1)
                                N = len(total_paths)
                                count = 0
                                for total_path in total_paths:
                                    count += len(total_path)
                                if count == 0:
                                    continue
                                total_path = None
                                while not total_path:
                                    choice = random.randint(0, N-1)
                                    M = len(total_paths[choice])
                                    if M > 0:
                                        choice1 = random.randint(0, M-1)
                                        total_path = total_paths[choice][choice1]
                                in_line.append(str(start)+' R'+str(edge)+'\n')
                                out_path = ""
                                length = int(len(total_path)/2)
                                for t in range(length):
                                    if t > 0:
                                        out_path += ' '
                                    out_path += ('R'+str(total_path[2*t])+' '+str(total_path[2*t+1]))
                                out_path += '\n'
                                out_line.append(out_path)
                    connected[start][end].append(edge)
        return (in_line, out_line)
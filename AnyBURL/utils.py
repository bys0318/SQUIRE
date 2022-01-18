import torch
import os
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
import random
import itertools
import multiprocessing as mp
import time
from random import sample
from tqdm import tqdm
import argparse
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=False, action="store_true")
    parser.add_argument("--path", default="path1/path-1000")
    parser.add_argument("--num", default=50, type=int) # threshold for appearing time, 50 for FB237
    parser.add_argument("--ratio", default=0.1, type=float) # threshold for true rate
    parser.add_argument("--dataset", default="FB15K237")
    args = parser.parse_args()
    return args

def filter_rule(path, thrshd_num, thrshd_ratio):
    rel2paths = dict()
    rel2scores = dict()
    rel2rules = dict()
    with open('../data/'+args.dataset+'/relation2id.txt', 'r') as f:
        for line in f.readlines():
            rel, N = line.strip().split('\t')
    N = int(N) + 1
    with open(path, 'r') as f:
        for line in f:
            meta_rule = line.strip().split('\t')
            num = int(meta_rule[0])
            ratio = float(meta_rule[2])
            rule = meta_rule[3]
            meta_rels = rule.strip().split()
            p = meta_rels[0]
            if num > thrshd_num and ratio > thrshd_ratio and p[-5:] == "(X,Y)":
                pos = p.find('(')
                p = int(p[1:pos])
                rels = meta_rels[2:]
                rel_path = []
                l = len(rels)
                last = 'X'
                for i in range(l):
                    rel = rels[i]
                    pos = rel.find('(')
                    r = int(rel[1:pos])
                    if rel[pos+1] == last:
                        last = rel[pos+3]
                    else:
                        assert last == rel[pos+3]
                        last = rel[pos+1]
                        r += N
                    rel_path.append(r)
                if p not in rel2paths:
                    rel2paths[p] = []
                    rel2scores[p] = []
                rel2paths[p].append(rel_path)
                score = ratio # can combine num into calculation of score
                rel2scores[p].append(score)
                # add reverse relation p_rev
                p_rev = p + N
                rel_path_rev = []
                for i in range(l-1, -1, -1):
                    r = rel_path[i]
                    if r >= N:
                        r -= N
                    else:
                        r += N
                    rel_path_rev.append(r)
                if p_rev not in rel2paths:
                    rel2paths[p_rev] = []
                    rel2scores[p_rev] = []
                rel2paths[p_rev].append(rel_path_rev)
                rel2scores[p_rev].append(score)
                
    for k in rel2paths.keys():
        rel2rules[k] = sorted(zip(rel2scores[k], rel2paths[k]), key=lambda pair: pair[0], reverse=True)
    with open('../data/'+args.dataset+'/rules.dict', 'w') as f:
        json.dump(rel2rules, f)

if __name__ == "__main__":
    args = get_args()
    filter_rule(args.path, args.num, args.ratio)

#!/user/bin/env python
# coding=utf-8
'''
@project : TransE
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : test.py
@ide     : PyCharm
@time    : 2022-07-01

@Description:
'''
import os
# import data
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

letter = [c for c in "abcdefghijklmnopqrstuvwxyz"]
letter_id = {c:i for (c, i) in zip(letter, range(26))}
id_letter = {i:c for (c, i) in zip(letter, range(26))}

emb = nn.Embedding(26, 5)
sentence = "this is my little kitty and she is really cute"
sentence.capitalize()
sen_list = [c for c in sentence if c != " "]
sen_id_list = [letter_id[c] for c in sen_list]
data = torch.from_numpy(np.array(sen_id_list)).long()
data_emb = emb(data)


pos_set = []
for o in range(1, 25):
    for i in range(25 - o):
        pos_set.append((i, 25 - o, i+1))

neg_set = []
cnt = 0
while(1):
    h = random.randint(0, 25)
    t = random.randint(0, 25)
    r = random.randint(1, 25)
    triple = (h, r, t)
    if triple not in pos_set:
        neg_set.append(triple)
        cnt += 1
    if cnt >= 1000:
        break

label = [True] * len(pos_set) + [False] * len(neg_set)
data = pos_set + neg_set


class Trainset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data_i = self.data[index]
        label_i = self.label[index]

        return data_i, label_i

    def __len__(self):
        return len(self.data)

MyDataset = Trainset(data, label)


TrainLoader = DataLoader(dataset=MyDataset, batch_size=4, shuffle=True)


from model import entity_embedding, relation_embedding


ent_emb = entity_embedding(26, 5)
rel_emb = relation_embedding(25, 5)


for (data, label) in TrainLoader:

    data, label = next(iter(TrainLoader))

    data = torch.concat(data, axis=1)
    h = data[0]
    r = data[1]
    t = data[2]

    h_emb = ent_emb(h)
    r_emb = rel_emb(r)
    t_emb = ent_emb(t)

    distance = (h_emb + r_emb - t_emb).norm(p=2, dim=1)
    criterion = nn.MarginRankingLoss(margin=1.0, reduction='none')

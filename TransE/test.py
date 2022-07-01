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
        pos_set.append((i, "A"+str(25 - o), i+1))

neg_set = []
cnt = 0
while(1):
    h = random.randint(0, 25)
    t = random.randint(0, 25)
    r = "A" + str(random.randint(1, 25))
    triple = (h, r, t)
    if triple not in pos_set:
        neg_set.append(triple)
        cnt += 1
    if cnt >= 1000:
        break




for i in range(21):
    pos_A_set.append((i, "A1", i+5))


train_path = os.path.join(path, "train.txt")
validation_path = os.path.join(path, "valid.txt")

train_set = data.FB15KDataset(train_path, entity2id, relation2id)
train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)



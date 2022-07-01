#!/user/bin/env python
# coding=utf-8
'''
@project : TransE
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : model.py
@ide     : PyCharm
@time    : 2022-07-01

@Description:
'''

import numpy as np
import torch
import torch.nn as nn



class entity_embedding(nn.Module):

    def __init__(self, word_cnt, dim):
        super(entity_embedding, self).__init__()

        self.word_cnt = word_cnt
        self.dim = dim
        self.ent_emb = nn.Embedding(num_embeddings=word_cnt+1,
                                    embedding_dim=dim,
                                    padding_idx=word_cnt)
        uniform_range = 6 / np.sqrt(self.dim)
        self.ent_emb.weight.data.uniform_(-uniform_range, uniform_range)

    def forward(self, x):
        return self.ent_emb(x)


class relation_embedding(nn.Module):

    def __init__(self, word_cnt, dim):
        super(relation_embedding, self).__init__()

        self.word_cnt = word_cnt
        self.dim = dim
        self.rel_emb = nn.Embedding(num_embeddings=word_cnt + 1,
                                    embedding_dim=dim,
                                    padding_idx=word_cnt)
        uniform_range = 6 / np.sqrt(self.dim)
        self.rel_emb.weight.data.uniform_(-uniform_range, uniform_range)
        self.rel_emb.weight.data[:-1, :].div_(self.rel_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))

    def forward(self, x):
        return self.ent_emb(x)



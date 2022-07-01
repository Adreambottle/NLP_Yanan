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
import data

train_path = os.path.join(path, "train.txt")
validation_path = os.path.join(path, "valid.txt")

train_set = data.FB15KDataset(train_path, entity2id, relation2id)
train_generator = torch_data.DataLoader(train_set, batch_size=batch_size)
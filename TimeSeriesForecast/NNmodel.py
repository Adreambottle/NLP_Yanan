#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : NNmodel.py
@ide     : PyCharm
@time    : 2022-06-13

@Description:
'''


import torch
import torch.nn as nn

class CNN_m(nn.Module):
    def __init__(self):
        super(CNN_m, self).__init__()
        self.Linear1 = nn.Linear(5, 8)
        self.Conv1d = nn.Conv1d(1, 3, kernel_size=(5,))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.Linear2 = nn.Linear(12, 1)

    def forward(self, x):
        x = self.Linear1(x)   # (B, 1, 5) - (B, 1, 8)

        x = self.Conv1d(x)    # (B, 1, 8) - (B, 3, 4)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.flatten()       # (B, 3, 4) - (B, 12)

        x = self.Linear2(x)   # (B, 12) - (B, 1)
        x = self.relu(x)
        return x


class MLP_m(nn.Module):
    def __init__(self):
        super(MLP_m, self).__init__()
        self.Linear1 = nn.Linear(5, 3)
        self.Linear2 = nn.Linear(3, 1)

        self.Sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.Linear1(x)       # (B, 1, 5) - (B, 1, 3)
        x = self.Sigmoid(x)
        x = self.dropout(x)
        x = self.Linear2(x)       # (B, 1, 3) - (B, 1, 1)
        x = self.Sigmoid(x)
        return x



class LSTM_m(nn.Module):
    def __init__(self):
        super(LSTM_m, self).__init__()
        self.in_dim = 5
        self.n_layer = 2
        self.hidden_dim = 10
        self.lstm = nn.LSTM(
            self.in_dim,
            self.hidden_dim,
            self.n_layer,
        )
        self.Linear = nn.Linear(self.hidden_dim, 1)
        self.Sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        # Get the last value of the hidden layer
        x = h_n[-1, :, :]
        x = self.Linear(x)
        return x


# x = torch.randn(1, 1, 5)
# M1 = LSTM_m()
# M2 = MLP_m()
# M1(x)
# M2(x)
# M2(series_new)
#
# x
# series_new
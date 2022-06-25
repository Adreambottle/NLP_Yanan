#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : LSTM.py
@ide     : PyCharm
@time    : 2022-06-14

@Description:
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from Toolkits import data_split, get_scaler


class LSTMModel():

    def __init__(self):

        self.data = None
        self.epochs = 0
        self.X_train = None
        self.Y_train = None
        self.look_back = 5
        self.scaler = None


    def fit(self, data):
        """
        Fit the LSTM model
        :param trainX:
        :param trainY:
        :return:
        """

        self.data = data
        Scaler = get_scaler(self.data)
        self.scaler = Scaler

        series_l = {}
        series_X = []
        series_Y = []

        data_columns = data.columns
        for label in data_columns:
            # label = "V1"
            series = data[label].dropna()
            # series = data[label].dropna()

            series = np.array(series).reshape(-1, 1)
            series = series.astype('float32')

            series = Scaler.transform(series)
            series_l[label] = series
            # series_l.append(series)
            trainX, trainY, testX, testY = data_split(series,
                                                      look_back=self.look_back,
                                                      train_pct=1)
            series_X.append(trainX)
            series_Y.append(trainY)

        X_train = np.concatenate(series_X, axis=0)
        Y_train = np.concatenate(series_Y, axis=0)

        self.X_train = X_train
        self.Y_train = Y_train


    def train(self, epochs=20, dropout=0.1):

        X_train = self.X_train
        Y_train = self.Y_train


        # Create LSTM Model using Keras
        model = Sequential()
        Layer1 = LSTM(units=20,
                      input_shape=(X_train.shape[1], X_train.shape[2]),
                      dropout=dropout)
        model.add(Layer1)
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, Y_train, epochs=epochs, batch_size=1, verbose=2)

        self.model = model



    def predict(self, pred_data, ):

        look_back = self.look_back
        model = self.model
        scaler = self.scaler


        # predict the value using the loop
        series_r = {}
        for label in pred_data:
            ser = pred_data[label]
            ser = np.array(ser)
            ser.astype('float32')
            ser = ser.flatten()
            for _ in range(8):
                series_new = ser[-look_back:]
                # print(f"label {label}, item {_}, {series_new}")
                pred_value = model.predict(series_new.reshape(1, 1, -1))
                ser = np.append(ser, pred_value)

            # inverse_transform to get the original value
            ser_predict = scaler.inverse_transform(ser.reshape(1, -1))
            ser_predict = ser_predict.flatten()
            series_r[label] = ser_predict

        # print(series_r)
        result_data = pd.DataFrame(series_r)
        return result_data





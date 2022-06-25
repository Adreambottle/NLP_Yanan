#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : LSTM_Simulate.py
@ide     : PyCharm
@time    : 2022-06-13

@Description:
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.utils.vis_utils import plot_model


# Function of creating the dataset
def moving_window_split(ser, look_back=3):
    """
    Moving window splitting method [x1, x2, x3] => x4
    splitting a series into dataX and labelY
    :param ser:
    :param look_back: the width of the window
    :return:
    """
    dataX, dataY = [], []
    for i in range(len(ser) - look_back - 1):
        a = ser[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(ser[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def data_split(series, look_back=3, train_pct=0.67, ):
    """
    Split a series into train data and test data
    Using the moving_window splitting method
    :param series: the original series
    :param look_back: the width of the moving window
    :param train_pct: percentage of the train part
    :return: trainX, trainY, testX, testY
    """

    # Split the the data into train and test two parts
    train_size = int(len(series) * train_pct)
    train, test = series[0:train_size, :], series[train_size:len(series), :]

    # Create train  and test look_back data
    trainX, trainY = moving_window_split(train, look_back)
    testX, testY = moving_window_split(test, look_back)

    # Change the data formation
    trainX = np.reshape(trainX, (trainX.shape[0], -1, trainX.shape[1]))  # （样本个数，1，输入的维度）
    if train_pct != 1:
        testX = np.reshape(testX, (testX.shape[0], -1, testX.shape[1]))
    else:
        testX = None
    return trainX, trainY, testX, testY



def fit_model(trainX, trainY, epoch, dropout):
    """
    Fit the LSTM model
    :param trainX:
    :param trainY:
    :return:
    """
    # Create LSTM Model using Keras
    model = Sequential()
    Layer1 = LSTM(units=20,
                  input_shape=(trainX.shape[1], trainX.shape[2]),
                  dropout=dropout)
    model.add(Layer1)
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epoch, batch_size=1, verbose=2)

    return model



def data_process(ser):

    # Read the data
    # ser = np.array(data_diff["V1"]).reshape(-1, 1)
    ser = ser.astype('float32')

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    ser = scaler.fit_transform(ser)


    trainX, trainY, testX, testY = data_split(ser, look_back=3, train_pct=0.67, )


    model = fit_model(trainX, trainY, epoch=20, dropout=0.1)


    # Predict
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # inverse_transform to get the original value
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Calculate the performance of the regression
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))


def main():
    # series = np.array(data_diff["V1"]).reshape(-1, 1)
    pass



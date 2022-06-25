#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : Toolkits.py
@ide     : PyCharm
@time    : 2022-06-14

@Description:
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# difference or division
def difference(ser, method="Div"):
    """
    calculate the difference or the division one by one
    :param ser: The original series
    :param method: difference or division
    :return:
    """
    diff_ser = []
    for i in range(len(ser) - 1):
        if method == "Diff":
            diff_ser.append(ser[i+1] - ser[i])
        if method == "Div":
            diff_ser.append(ser[i+1] / ser[i])

    return diff_ser



def recover(diff_ser, start_value, method="Div", ):
    """
    Recover from the or
    :param diff_ser:
    :param start_value: add the first value
    :param method:
    :return:
    """
    diff_ser = diff_ser.dropna()
    diff_ser = np.append(np.array([start_value]), np.array(diff_ser))

    if method == "Diff":
        ser = diff_ser.cumsum()
    elif method == "Div":
        ser = diff_ser.cumprod()
    else:
        raise TypeError("You should only use Diff or Div")

    return ser


def plot_data(data, title: str):
    plt.figure(figsize=(10, 5))

    labels = ["V1", "V2", "V3", "V4"]
    colors = ["r-", "b-", "g-", "y-"]
    for (i, label) in enumerate(labels):
        plt.plot(data.index, data[label], colors[i], marker="o", label=label)

    plt.legend(loc="best")
    plt.title(title, fontweight='heavy')

    plt.show()

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



def get_scaler(data):
    """
    Get the scaler for the total data
    Combine all the data in a series
    :return: the total_scaler
    """

    ser_to_concat = []

    for label in data.columns:
        # ser = data[label].dropna()
        ser = data[label].dropna()
        ser_to_concat.append(ser)

    ser_total = pd.concat(ser_to_concat, axis=0)
    ser_total = np.array(ser_total).reshape(-1, 1)

    Scaler = MinMaxScaler(feature_range=(0, 1))
    Scaler.fit(ser_total)
    return Scaler


def generate_datetime():
    time_series_20 = [pd.to_datetime(f"2020-{month}") for month in range(1, 13)]
    time_series_21 = [pd.to_datetime(f"2021-{month}") for month in range(1, 13)]
    time_series = time_series_20 + time_series_21


class Model(object):
    """
    基本模型，要求所有实现的模型需要具有以下基本组件
    """

    def __init__(self):
        pass

    def fit(self, **kwargs):
        raise NotImplementedError
        pass

    def train(self, **kwargs):
        raise NotImplementedError
        pass

    def predict(self, **kwargs):
        raise NotImplementedError
        pass

    def save_model(self, **kwargs):
        raise NotImplementedError
        pass

    def restore_model(self, **kwargs):
        raise NotImplementedError
        pass
#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : NeuralNetwork.py
@ide     : PyCharm
@time    : 2022-06-13

@Description:
'''


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data

from Toolkits import data_split, get_scaler, Model
from NNmodel import MLP_m, CNN_m, LSTM_m




class NNModel(Model):

    def __init__(self, model_name="MLP"):
        super(NNModel).__init__()
        self.model_name = model_name
        self.data = None
        self.epochs = None
        self.look_back = 5

        if model_name == "MLP":
            self.model = MLP_m()
        if model_name == "CNN":
            self.model = CNN_m()
        if model_name == "LSTM":
            self.model = LSTM_m()



    def set_model(self, model_name):
        self.model_name = model_name
        if model_name == "MLP":
            self.model = MLP_m()
        if model_name == "CNN":
            self.model = CNN_m()
        if model_name == "LSTM":
            self.model = LSTM_m()


    def fit(self, data):

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
            trainX, trainY, testX, testY = data_split(series, look_back=5, train_pct=1)
            series_X.append(trainX)
            series_Y.append(trainY)

        X_total = np.concatenate(series_X, axis=0)
        Y_total = np.concatenate(series_Y, axis=0)

        X_total = torch.from_numpy(X_total.astype(np.float32))
        Y_total = torch.from_numpy(Y_total.astype(np.float32))

        train_data = Data.TensorDataset(X_total, Y_total)
        train_loader = Data.DataLoader(dataset=train_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=4,
                                       drop_last=False)

        self.train_loader = train_loader

    # model = MLP()
    # model = CNNnetwork()


    def train(self, epochs=20):
        """
        Train the model by
        :param epochs:
        :return:
        """
        print("\nTraining the NN model")

        model = self.model
        torch.manual_seed(101)
        train_loader = self.train_loader

        # Set the loss function and the optimizer
        # Using MSE as Loss and Adam as optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()

        best_epoch = 0
        best_loss = 1
        best_model = None
        for epoch in range(epochs):

            loss = 0
            for X, y_real in train_loader:
                optimizer.zero_grad()
                y_pred = model(X.reshape(1, 1, -1)).flatten()
                # print(y_pred)
                loss = criterion(y_pred, y_real)
                loss.backward()
                optimizer.step()

            if loss < best_loss:
                best_epoch = epoch
                best_loss = loss
                best_model = model

            print(f'Epoch: {epoch}, Loss: {loss.item():10.8f}')

        print(f'Best Epoch: {best_epoch}, Best Loss: {best_loss.item():10.8f}')

        return best_model


    def predict(self, pred_data):

        look_back = self.look_back
        model = self.model
        scaler = self.scaler
        model.eval()


        # predict the value using the loop
        series_r = {}
        for label in pred_data:
            ser = pred_data[label]
            ser = np.array(ser)
            ser = ser.flatten()
            for _ in range(8):
                series_new = ser[-look_back:]
                series_new = torch.from_numpy(series_new)
                series_new = series_new.to(torch.float32)
                series_new = series_new.reshape(1, 1, -1)
                pred_value = model(series_new)
                ser = np.append(ser, pred_value.item())


            # inverse_transform to get the original value
            ser_predict = scaler.inverse_transform(ser.reshape(1, -1))
            ser_predict = ser_predict.flatten()
            series_r[label] = ser_predict

        # print(series_r)
        result_data = pd.DataFrame(series_r)
        print("\nFinished Predict")

        return result_data




# def main():
#
#     data = data_div[["V1", "V2", "V4"]]
#     DataModel = NNModel()
#     DataModel.set_model("MLP")
#     DataModel.fit(data)
#     DataModel.train(epochs=10)
#     result_data = DataModel.predict(data)
#
#     DataModel.model()
#     look_back = DataModel.look_back
#     model = DataModel.model
#     scaler = DataModel.scaler

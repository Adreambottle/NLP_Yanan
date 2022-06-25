#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : ARIMA.py
@ide     : PyCharm
@time    : 2022-06-14

@Description:
'''

import pandas as pd
from pmdarima import auto_arima
from Toolkits import Model


class ArimaModel(Model):

    def __init__(self):
        super(ArimaModel).__init__()
        self.data = None
        self.model_dict = {}
        self.predict_column = None
        self.ds = None

    def fit(self, data, ds):
        # fit the model
        # define the model
        self.data = data
        self.ds = ds

    def train(self):
        """
        # Create a model dict to store models with difference columns
        # This is the auto-arima model, which can find most suitable p, q, i automatically
        # The searching range of p is [1, 3]
        # The searching range of q is [1, 3]

        """

        print("\nTraining the auto_arima model")
        data = self.data

        model_dict = {}
        for label in data.columns:
            print(f"\nTraining the auto_arima model for column {label}")

            ser = data[label].dropna()
            model = auto_arima(ser,
                               start_p=1,
                               start_q=1,
                               max_p=3,
                               max_q=3,
                               start_P=0,
                               seasonal=True,
                               d=1,
                               D=0,
                               trace=True,
                               error_action='ignore',
                               suppress_warnings=False)
            model.fit(ser)
            model_dict[label] = model

        # Store the model as a class attribute
        self.model_dict = model_dict
        print("\nFinished Train")


    def predict(self, data):
        """
        Predict the next 8 value, based on the input data
        """
        result_data = []

        for label in data.columns:

            model = self.model_dict[label]
            predict_value = model.predict(n_periods=8)

            predict_value = pd.Series(predict_value)
            predict_value = predict_value.rename(label)
            result_data.append(predict_value)

        result_data = pd.concat(result_data, axis=1)
        result_data = pd.concat([data, result_data], axis=0)
        result_data.reset_index(drop=True, inplace=True)
        print("\nFinished Predict")
        return result_data

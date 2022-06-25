#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : Prophet.py
@ide     : PyCharm
@time    : 2022-06-13

@Description: Using Prophet method to predict the data
'''


import pandas as pd
from prophet import Prophet
from Toolkits import Model


class ProphetModel(Model):

    def __init__(self):
        super(ProphetModel).__init__()
        self.data = None
        self.model_dict = {}
        self.predict_column = None
        self.ds = None


    def fit(self, data, ds):
        """
        Fit the prophet model with the data
        The propeht data needs two inputs, time index and data value
        :param data: The value of the data
        :param ds: Date index
        :return:
        """
        # fit the model
        # define the model
        self.data = data
        self.ds = ds

    def train(self):
        """
        Train the Prophet Model
        :return:
        """

        print("\nTraining the Prophet model")
        data = self.data
        ds = self.ds

        # Create a model dict to store models with difference columns
        model_dict = {}
        for label in data.columns:
            print(f"\nTraining the Prophet model for column {label}")

            df = pd.concat([ds, data[label]], axis=1)
            df.columns = ['ds', 'y']

            model = Prophet()
            model.fit(df)
            model_dict[label] = model

        # Store the model as a class attribute
        self.model_dict = model_dict

        print("\nFinished Train")


    def predict(self, data):

        # Build the future time index to predict
        time_to_pred = [pd.to_datetime(f"2021-{month}") for month in range(5, 13)]

        result_data = []
        for label in data.columns:

            # Load the model from model_dict
            model = self.model_dict[label]
            data_to_pred = pd.DataFrame(time_to_pred, columns=["ds"])

            # Only select the trend column as the predicted value
            predict_value = model.predict(data_to_pred)["trend"]

            predict_value = predict_value.rename(label)
            result_data.append(predict_value)

        result_data = pd.concat(result_data, axis=1)
        result_data = pd.concat([data, result_data], axis=0)
        result_data.reset_index(drop=True, inplace=True)
        print("\nFinished Predict")

        return result_data

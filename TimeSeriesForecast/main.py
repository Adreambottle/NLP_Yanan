#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : main.py
@ide     : PyCharm
@time    : 2022-06-13

@Description:
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from Toolkits import difference, recover
import warnings

warnings.filterwarnings('ignore')


# Processing the data
data = pd.read_csv("Cohorts_rev.csv")
org_columns = list(data.columns)
labels = ["V1", "V2", "V3", "V4"]
data.columns = ["Date"] + labels

# Change date into datetime formate
data["Date"] = pd.to_datetime(data["Date"], format="%b-%y")
data_org = data.copy()

# View plot and test the stability of the data
def test_stable(ser):
    adf_result = adfuller(ser)
    print(f"The result of ADF Test:\n  t-statistic: {adf_result[0]}\n  p-vale: {adf_result[1]}")

def plot_data(data, title):

    plt.figure(figsize=(10, 5))
    plt.style.use('seaborn-white')
    labels = ["V1", "V2", "V3", "V4"]
    colors = ["r-", "b-", "g-", "y-"]
    for (i, label) in enumerate(labels):
        plt.plot(data.index, data[label], colors[i], marker="o", label=label)
    plt.legend(loc="best")
    plt.title = title
    plt.show()


# View the original data
plot_data(data_org, "Original Data")
test_stable(data_org["V1"])


# Use log to make the data linearly
data_log = data.copy()
for label in labels:
    data_log[label] = np.log(data[label] + 1)

plot_data(data_log, "Log Data")
test_stable(data_log["V1"].dropna())


# Use difference to get the stable data
data_diff = data.copy()
for label in labels:
    data_diff[label] = data[label].diff()

plot_data(data_diff, "Difference Data")
test_stable(data_diff["V1"].dropna())



# Use division as difference
data_div = data.copy()
for label in labels:
    data_div[label] = np.append([np.nan], difference(data[label], "Div"))

plot_data(data_div, "Division Data")
test_stable(data_div["V1"].dropna())




# Series Decomposition Analysis

from statsmodels.tsa.seasonal import seasonal_decompose

data_V3 = data["V3"]
data_V3.index = data["Date"]

decompose_result = seasonal_decompose(data_V3.dropna(),
                                      model='multiplicative',
                                      period=6)
decompose_result.plot()
plt.show()


# y(t) = Level + Trend + Seasonality + Noise
# Remove the influence of Seasonality

data_dec = data.copy()
data_dec.set_index("Date", inplace=True)
seasonality = pd.Series(decompose_result.seasonal)
data_dec["V3"] = data_dec["V3"] - seasonality
plot_data(data_dec, "Remove Seasonality")
test_stable(data_dec["V1"].dropna())

# It is not good, because the value cannot less than zero
print(data_dec["V3"])


# Replace the peak value with the average value
# The index of the peak value is 7 and 13
# The next index of next peak is 19

from collections import defaultdict
data_chg = data.copy()

# Store the trend and seasonality in a dictionary
decompose_dict = defaultdict()
decompose_dict[7] = defaultdict()
decompose_dict[13] = defaultdict()
decompose_dict[19] = defaultdict()
decompose_dict[7]["trend"] = (data_chg["V3"][6] + data_chg["V3"][8]) / 2
decompose_dict[13]["trend"] = (data_chg["V3"][12] + data_chg["V3"][14]) / 2
decompose_dict[7]["seasonality"] = data_chg["V3"][7] - decompose_dict[7]["trend"]
decompose_dict[13]["seasonality"] = data_chg["V3"][13] - decompose_dict[13]["trend"]

# Replace the data of peak by its trend
data_chg["V3"][7] = decompose_dict[7]["trend"]
data_chg["V3"][13] = decompose_dict[13]["trend"]

print(data_chg["V3"])

print(f'The seasonality of 7 is:  {decompose_dict[7]["seasonality"] }')
print(f'The seasonality of 13 is: {decompose_dict[13]["seasonality"] }')


shrinkrage = decompose_dict[7]["seasonality"] / decompose_dict[13]["seasonality"]
decompose_dict[19]["seasonality"] = decompose_dict[13]["seasonality"] / shrinkrage
print(f'The seasonality of 19 is: {decompose_dict[19]["seasonality"] }')

plot_data(data_chg, "Data Changed")


# Use division as difference
data_use = data_chg.copy()
data_use.to_csv("data_use.csv")
for label in labels:
    data_use[label] = np.append([np.nan], difference(data_use[label], "Div"))

plot_data(data_use, "Data in Use")




from LSTM import LSTMModel
from NeuralNetwork import NNModel
from Prophet import ProphetModel
from ARIMA import ArimaModel
# from ClassicModel import




data = data_use[["V1", "V2", "V3", "V4"]]
date_index = data_use["Date"]


DataModel = LSTMModel()

# DataModel = NNModel("MLP")
DataModel.fit(data)
DataModel.train(epochs=10)
result_data = DataModel.predict(data)


# Prophet
DataModel = ProphetModel()
DataModel.fit(data, date_index)
DataModel.train()
result_data = DataModel.predict(data)



recover_data = {}
for label in result_data.columns:
    # The start value is the last valid data in the original data on 2021-04-01
    start_value = np.array(data_org[label].dropna())[-1]

    # We only use the data after 2021-04-01, the last 8 data
    ser_before = result_data[label][-8:]

    # recover the original data from the division data
    ser_after = recover(ser_before, start_value, "Div")

    # remove the first original value
    recover_data[label] = ser_after[1:]

# Concat the original data and the predicted data together
recover_data["Date"] = [pd.to_datetime(f"2021-{month}") for month in range(5, 13)]
recover_frame = pd.DataFrame(recover_data)
final_data = pd.concat([data_org, recover_frame])
final_data.reset_index(drop=True, inplace=True)

print(final_data)

# Add the seasonality for V3
final_data["V3"][19] = final_data["V3"][19] + decompose_dict[19]["seasonality"]
final_data.columns = org_columns
plot_data(final_data, "final_data")
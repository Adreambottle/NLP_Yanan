#!/user/bin/env python
# coding=utf-8
'''
@project : Liquidity
@author  : Daniel Yanan ZHOU (周亚楠)
@contact : adreambottle@outlook.com
@file    : Classic.py
@ide     : PyCharm
@time    : 2022-06-14

@Description:
'''

"""
"knn"     sklearn.neighbors.KNeighborsRegressor as Knn
"rf"      sklearn.ensemble.RandomForestClassifier
"ada"     sklearn.ensemble.AdaBoostRegressor
"gbrt"    sklearn.ensemble.GradientBoostingRegressor
"svr"     sklearn.svm.SVR
"lasso"   sklearn.linear_model.Lasso
"decision tree"       sklearn.tree.DecisionTreeRegressor
"linear"  sklearn.linear_model.LinearRegression
"ridge"   sklearn.linear_model.Ridge
"""

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as Knn
import pandas as pd
import numpy as np
from itertools import product
from Toolkits import data_split


regression_model_list = ["knn", "rf", "ada", "gbrt", "svr", "lasso", "decision tree", "linear", "ridge", "enet"]



def get_para(model_type, feature_num):

    svr_para_dict = [
        {"C": [0.1, 1, 10, 100], "kernel": ['linear'], "epsilon": [0.01, 0.05, 0.1, 0.5, 1]},
        {"C": [0.1, 1, 10, 100], "kernel": ['poly'], "degree": [2, 3, 4],
         "gamma": [1 / feature_num, 1e-2, 1e-3, 10], "epsilon": [0.01, 0.05, 0.1, 0.5, 1]},
        {"C": [0.1, 1, 10, 100], "kernel": ['rbf', 'sigmoid'], "gamma": [1 / feature_num, 1e-2, 1e-3, 10],
         "epsilon": [0.01, 0.05, 0.1, 0.5, 1]}
    ]

    gbdt_r_para_dict = [
        {"loss": ['ls', 'lad', 'huber', 'quantile'], "learning_rate": [0.01, 0.1, 1],
         "n_estimators": [20, 40, 60, 80, 100, 120, 140], "max_depth": [2, 3, 4],
         "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3],"random_state":[42]}
    ]

    rf_r_para_dictt = [
        {"criterion": ["mse", "mae"], "n_estimators": [5, 10, 15, 20], "max_features": ['sqrt', None],
         "min_samples_split": [2, 3, 4], "min_samples_leaf": [1, 2, 3], "random_state": [42],
         "bootstrap":[False]}
    ]

    dt_r_pata_dict = [
        {"criterion": ["mse", "mae"], "min_samples_split": [2, 3, 4, 5], "min_samples_leaf": [1, 2, 3, 4],
         "random_state": [42]}
    ]

    knn_r_para_dict = [{"n_neighbors": [3, 4, 5, 6, 7, 8], "p": [1, 2], "weights": ["uniform", "distance"]}]

    ada_r_para_dict = [{"n_estimators": [10, 20, 30, 40, 50, 60, 70, 80], "learning_rate": [0.1, 0.5, 1.0, 1.5, 2.0],
                        "loss": ['linear', 'square', 'exponential'], "random_state": [42]}]

    lasso_para_dict = [{"alpha": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]}]

    linear_para_dict = [{"normalize": [True, False]}]

    ridge_para_dict = [{"alpha": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],"random_state": [42]}]

    enet_para_dict = [{"l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]


    if model_type == 'knn':
        return knn_r_para_dict
    elif model_type == 'rf':
        return rf_r_para_dictt
    elif model_type == 'ada':
        return ada_r_para_dict
    elif model_type == 'gbrt':
        return gbdt_r_para_dict
    elif model_type == 'svr':
        return svr_para_dict
    elif model_type == 'lasso':
        return lasso_para_dict
    elif model_type == 'decision tree':
        return dt_r_pata_dict
    elif model_type == 'linear':
        return linear_para_dict
    elif model_type == 'ridge':
        return ridge_para_dict
    elif model_type == 'enet':
        return enet_para_dict
    else:
        raise ValueError("The model in not in the model list, you can add a new one!")

    pass


def get_base_model(model_type, para=None):
    if model_type == 'knn':
        if para is None:
            model = Knn()
        else:
            model = Knn(**para)

    elif model_type == 'rf':
        if para is None:
            model = RandomForestRegressor(random_state=42, bootstrap=False)
        else:
            model = RandomForestRegressor(**para)

    elif model_type == 'ada':
        if para is None:
            model = AdaBoostRegressor(random_state=42)
        else:
            model = AdaBoostRegressor(**para)

    elif model_type == 'gbrt':
        if para is None:
            model = GradientBoostingRegressor(random_state=42)
        else:
            model = GradientBoostingRegressor(**para)

    elif model_type == 'svr':
        if para is None:
            model = SVR()
        else:
            model = SVR(**para)

    elif model_type == 'lasso':
        if para is None:
            model = Lasso()
        else:
            model = Lasso(**para)

    elif model_type == 'decision tree':
        if para is None:
            model = DecisionTreeRegressor(random_state=42)
        else:
            model = DecisionTreeRegressor(**para)

    elif model_type == 'linear':
        if para is None:
            model = LinearRegression()
        else:
            model = LinearRegression(**para)

    elif model_type == 'ridge':
        if para is None:
            model = Ridge(random_state=42)
        else:
            model = Ridge(**para)

    elif model_type == 'enet':
        if para is None:
            model = ElasticNet()
        else:
            model = ElasticNet(**para)
    else:
        raise ValueError("The model in not in the model list, you can add a new one!")
    return model



def build_dataset(data, look_back=5):
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

        series_l[label] = series
        # series_l.append(series)
        trainX, trainY, testX, testY = data_split(series,
                                                  look_back=look_back,
                                                  train_pct=1)
        series_X.append(trainX)
        series_Y.append(trainY)

    X_train = np.concatenate(series_X, axis=0).squeeze()
    Y_train = np.concatenate(series_Y, axis=0)
    return X_train, Y_train


def predict(model, pred_data, look_back=5):
    series_r = {}
    for label in pred_data:
        ser = pred_data[label]
        ser = np.array(ser)
        ser.astype('float32')
        ser = ser.flatten()
        for _ in range(8):
            series_new = ser[-look_back:]
            print(f"label {label}, item {_}, {series_new}")
            pred_value = model.predict(series_new.reshape(1, -1))
            ser = np.append(ser, pred_value)

        # inverse_transform to get the original value
        ser_predict = ser.flatten()
        series_r[label] = ser_predict
    result_data = pd.DataFrame(series_r)
    return result_data



data_use = pd.read_csv("data_use")
data = data_use[["V1", "V2", "V3", "V4"]]
date_index = data_use["Date"]
LOOK_BACK = 5

X_train, Y_train = build_dataset(data=data, look_back=5)


# Take KNN as an example

model_type = "knn"
para_dict = get_para(model_type, 1)[0]
para_permuatabtion = [para_dict[key] for key in para_dict]

para_set = []
for sample in product(*para_permuatabtion):
    para_sample = {}
    for (i, key) in enumerate(para_dict):
        para_sample[key] = sample[i]
    para_set.append(para_sample)

print(f"Number of the parameter combinations {len(para_set)}")

for para_sample in para_set:
    print(para_sample)


result_data_list = []
for i, para_sample in enumerate(para_set):

    model = get_base_model(model_type, para_sample)
    model.fit(X_train, Y_train)
    result_data = predict(model, data, LOOK_BACK)
    result_data_list.append(result_data)


# Take decision tree as an example
model_type = "decision tree"
para_dict = get_para(model_type, 1)[0]
para_permuatabtion = [para_dict[key] for key in para_dict]

para_set = []
for sample in product(*para_permuatabtion):
    para_sample = {}
    for (i, key) in enumerate(para_dict):
        para_sample[key] = sample[i]
    para_set.append(para_sample)

print(f"Number of the parameter combinations {len(para_set)}")

for para_sample in para_set:
    print(para_sample)


result_data_list = []
for i, para_sample in enumerate(para_set):

    model = get_base_model(model_type, para_sample)
    model.fit(X_train, Y_train)
    result_data = predict(model, data, LOOK_BACK)
    result_data_list.append(result_data)
# -*- coding: utf-8 -*-

# @Time    : 2019-04-26 16:19
# @Author  : jian
# @File    : find_optimal_param.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve


def validation_curve_demo(x, y, model, param_name, param_range):
    """

    :param x:
    :param y:
    :param model:
    :param param_name:
    :param param_range:
    :return: 输出一个图像，用来选取最佳参数值
    """

    train_loss, test_loss = validation_curve(
        model, x, y, param_name=param_name,
        param_range=param_range, cv=5,
        scoring='neg_mean_squared_error')
    # print(train_loss, test_loss)
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(param_range, train_loss_mean, 'o-', color='r',
             label='Training')
    plt.plot(param_range, test_loss_mean, 'o-', color='g',
             label='Cross-validation')

    plt.xlabel(param_name)
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')
param_range = [x for x in range(1, 30)]

x = df.iloc[:, :-1]
y = df.target

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

if __name__ == "__main__":
    validation_curve_demo(x_train, y_train.astype("int"), RandomForestRegressor(), "n_estimators", param_range)

# -*- encoding:utf-8 -*-
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import matplotlib.pylab as plt

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

x = df.iloc[:, :-1]
y = df.target

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

model = RandomForestRegressor(
    n_estimators=40,  # 学习器个数
    # criterion='mse',  # 评价函数
    max_depth=23,  # 最大的树深度，防止过拟合
    min_samples_split=20,  # 根据属性划分节点时，每个划分最少的样本数
    min_samples_leaf=5,  # 最小叶子节点的样本数，防止过拟合
    max_features='auto',  # auto是sqrt(features)还有 log2 和 None可选
    max_leaf_nodes=None,  # 叶子树的最大样本数
    bootstrap=True,  # 有放回的采样
    min_weight_fraction_leaf=0,
    n_jobs=5)  # 同时用多少个进程训练

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("随机森林 score", score)
y_pred = model.predict(x_test)
print("随机森林 mean_squared_error", mean_squared_error(y_test, y_pred))

from utils import validation_curve_demo


# param_range = [x for x in range(1, 100)]
# validation_curve_demo(x_train, y_train, model, "n_estimators", param_range)
# # 最优参数 40

param_range = [x for x in range(10, 40, 1)]
validation_curve_demo(x_train, y_train, model, "max_depth", param_range)
# 最优参数 6

# param_range = [x for x in range(2, 30)]
# validation_curve_demo(x_train, y_train, model, "min_samples_split", param_range)
# 最优参数 4 -6 都可以 提升不大

# param_range = [x for x in range(2, 30)]
# validation_curve_demo(x_train, y_train, model, "min_samples_leaf", param_range)
# 最优参数 4 -6 都可以 提升不大

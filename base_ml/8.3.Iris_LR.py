#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


path = '../data/8.iris.data'  # 数据文件路径
df = pd.read_csv(path, header=0)
x = df.values[:, :-1]
y = df.values[:, -1]

# 使用sklearn的数据预处理
# 处理标签
le = preprocessing.LabelEncoder()
le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
y = le.transform(y)

# 处理特征
# x = StandardScaler().fit_transform(x)

#模型
# lr = LogisticRegression()  # Logistic回归模型
# lr.fit(x, y.ravel())  # 根据数据[x,y]，计算回归参数


# 等价形式
lr = Pipeline([('sc', StandardScaler()),
               ('clf', LogisticRegression())])
lr.fit(x, y.ravel())


# 训练集上的预测结果
y_hat = lr.predict(x)
y = y.reshape(-1)
result = y_hat == y
print(y_hat)
print(result)
acc = np.mean(result)
print('准确度: %.2f%%' % (100 * acc))

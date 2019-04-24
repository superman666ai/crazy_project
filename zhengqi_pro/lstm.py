# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


train_path = 'data/zhengqi_train.txt'
test_path = 'data/zhengqi_test.txt'


def load_data(path):
    df = pd.read_csv(path, sep="\t")
    return df


train_df = load_data(train_path)
test_df = load_data(test_path)

x = train_df.iloc[:, :-1]
y = train_df.target

print(y)

# 标准化数据 minmax
x_mm = MinMaxScaler()
x = x_mm.fit_transform(x)
test_df = x_mm.transform(test_df)
y_mm = MinMaxScaler()
y = y_mm.fit_transform(y)
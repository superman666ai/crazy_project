# -*- coding: utf-8 -*-

# @Time    : 2019-03-19 16:21
# @Author  : jian
# @File    : nn_re.py
# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile, mutual_info_regression

train_path = 'data/zhengqi_train.txt'
test_path = 'data/zhengqi_test.txt'

# load data
def load_data(path):
    df = pd.read_csv(path, sep="\t")
    return df

train_df = load_data(train_path)
test_df = load_data(test_path)

x = train_df.iloc[:, :-1]
y = train_df.target.values
# print(x.shape)
# 特征处理
# x = x[['V0', 'V1', 'V2', 'V4', 'V8', 'V12', 'V27', 'V31', 'V37']]

# test_df = test_df[['V0', 'V1', 'V2', 'V4', 'V8', 'V12', 'V27', 'V31', 'V37']]

# 卡方 选择

model1 = SelectPercentile(mutual_info_regression, percentile=95)  # 选择k个最佳特征
x = model1.fit_transform(x, y)
test_df = model1.transform(test_df)
print(x)
print(model1.scores_)

y = np.array(y)[:, np.newaxis]



# # 特征处理 筛选特征
# clf = ExtraTreesRegressor()
# clf = clf.fit(x, y)
#
# model = SelectFromModel(clf, prefit=True)
# x = model.transform(x)
# test_df = model.transform(test_df)


# PCA过程
# pca = PCA(n_components=0.9)
# pca.fit(x)
# test_df = pca.transform(test_df)

# 标准化数据 minmax
x_mm = MinMaxScaler()
x = x_mm.fit_transform(x)
test_df = x_mm.transform(test_df)
y_mm = MinMaxScaler()
y = y_mm.fit_transform(y)
#
# print(y)

# # # 标准化数据 scaler
# x_mm = StandardScaler()
# x = x_mm.fit_transform(x)
# test_df = x_mm.transform(test_df)
#
# y_mm = StandardScaler()
# y = y_mm.fit_transform(y)


# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)


# 添加层

# 创建一个神经网络层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


xs = tf.placeholder(shape=[None, x_train.shape[1]], dtype=tf.float32, name="inputs")
ys = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y_true")
drop_out = 0.6

l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.sigmoid)
# l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.tanh)
# l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.relu)

l1 = layer1 = tf.nn.dropout(l1, drop_out)

l2 = add_layer(l1, 100, 10, activation_function=tf.nn.sigmoid)
# l2 = add_layer(l1, 100, 10, activation_function=tf.nn.tanh)
# l2 = add_layer(l1, 100, 10, activation_function=tf.nn.relu)

l2 = tf.nn.dropout(l2, drop_out)


prediction = add_layer(l2, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

feed_dict_train = {ys: y_train, xs: x_train}

# Start training

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Fit all training data
    for i in range(10000):
        sess.run(train_step, feed_dict=feed_dict_train)

        if i % 50 == 0:

            train_acc = sess.run(loss, feed_dict=feed_dict_train)
            print("TRAIN ACCURACY: %.3f" % (train_acc))

            feeds = {xs: x_test, ys: y_test}
            test_acc = sess.run(loss, feed_dict=feeds)
            print("TEST ACCURACY: %.3f" % (test_acc))

            # 保存结果
            # if float(test_acc) < 0.2:
            #     y_pre = sess.run(prediction, feed_dict={xs: test_df})
            #     y_pre = y_mm.inverse_transform(y_pre)
            #     pre = pd.DataFrame(y_pre, columns=["0"])
            #     pre = pre["0"]
                # save_result(list(pre))

"""
画出平面图形 
# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_train, y_train)
plt.ion()#本次运行请注释，全局运行不要注释
plt.show()


"""

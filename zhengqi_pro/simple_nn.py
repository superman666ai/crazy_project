# -*- coding: utf-8 -*-

# @Time    : 2019-04-29 15:55
# @Author  : jian
# @File    : simple_nn.py

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
from utils import save_result
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile, mutual_info_regression

from a import train_x_head2, Y, np, test2
test_df = test2
Y = np.array(Y)[:, np.newaxis]

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(train_x_head2, Y, test_size=0.25, random_state=1)


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

# l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.sigmoid)
l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.tanh)
# l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.relu)

l1 = layer1 = tf.nn.dropout(l1, drop_out)

# l2 = add_layer(l1, 100, 10, activation_function=tf.nn.sigmoid)
l2 = add_layer(l1, 100, 10, activation_function=tf.nn.tanh)
# l2 = add_layer(l1, 100, 10, activation_function=tf.nn.relu)

l2 = tf.nn.dropout(l2, drop_out)


prediction = add_layer(l2, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

feed_dict_train = {ys: y_train, xs: x_train}

# Start training
saver = tf.train.Saver()
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Fit all training data
    for i in range(1000000):
        sess.run(train_step, feed_dict=feed_dict_train)

        if i % 100 == 0:
            train_acc = sess.run(loss, feed_dict=feed_dict_train)
            print("TRAIN ACCURACY: %.3f" % (train_acc))

            feeds = {xs: x_test, ys: y_test}
            test_acc = sess.run(loss, feed_dict=feeds)
            print("TEST ACCURACY-----: %.3f" % (test_acc))


    saver_path = saver.save(sess, "save/model.ckpt")
    print("Model saved in file: ", saver_path)


with tf.Session() as sess:
    saver.restore(sess, "save/model.ckpt")
    y_pre = sess.run(prediction, feed_dict={xs: test_df})
    pre = pd.DataFrame(y_pre, columns=["0"])
    pre = pre["0"]
    save_result(list(pre))

    print ("end")


"""
画出平面图形 
# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_train, y_train)
plt.ion()#本次运行请注释，全局运行不要注释
plt.show()


"""

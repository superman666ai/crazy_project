# -*- coding: utf-8 -*-

# @Time    : 2019-04-08 8:49
# @Author  : jian
# @File    : activate_func.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# data

x = np.linspace(-7, 7, 180)


# activation func

def sigmoid(inputs):
    y = [1 / float(1 + np.exp(-x)) for x in inputs]
    return y


def relu(inputs):
    y = [x * (x > 0) for x in inputs]
    return y


def tanh(inputs):
    y = [((np.exp(x) - np.exp(-x)) / float(np.exp(x) + np.exp(-x))) for x in inputs]
    return y


def softplus(inputs):
    y = [np.log(1 + np.exp(x)) for x in inputs]
    return y

# 经过自定义处理y的值
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
y_softplus = softplus(x)




# 经过TensorFlow处理的Y值
# y_sigmoid = tf.nn.sigmoid(x)
#
# y_relu = tf.nn.relu(x)
#
# y_tanh = tf.nn.tanh(x)
#
# y_softplus = tf.nn.softplus(x)


# sess = tf.Session()
# y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])

# 创建图像
plt.subplot(221)
plt.plot(x, y_sigmoid, c="red", label="sigmoid")
plt.ylim(-0.2, 1.2)
plt.legend(loc="best")

plt.subplot(222)
plt.plot(x, y_relu, c="red", label="relu")
plt.ylim(-1, 6)
plt.legend(loc="best")

plt.subplot(223)
plt.plot(x, y_tanh, c="red", label="tanh")
plt.ylim(-1.3, 1.3)
plt.legend(loc="best")

plt.subplot(224)
plt.plot(x, y_softplus, c="red", label="softplus")
plt.ylim(-1, 6)
plt.legend(loc="best")

plt.show()

# sess.close()
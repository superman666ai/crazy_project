# -*- coding: utf-8 -*-

# @Time    : 2019-04-08 11:43
# @Author  : jian
# @File    : mnist_cnn_tf.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 导入数据
mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

# 从 Test（测试）数据集里选取 3000 个手写数字的图片和对应标签
test_x = mnist.test.images[:3000]  # 图片
test_y = mnist.test.labels[:3000]  # 标签

# 输入
input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.
output_y = tf.placeholder(tf.float32, [None, 10])
input_x_images = tf.reshape(input_x, [-1, 28, 28, 1])

# 卷积 28*28*1 28*28*32
conv1 = tf.layers.conv2d(input_x_images, filters=32, kernel_size=[5, 5], strides=1, padding="same",
                         activation=tf.nn.relu)

# 池化 28*28*32  14*14*32
pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

# 卷积 14*14*32  14*14*64
conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[5, 5], strides=1, padding="same", activation=tf.nn.relu)

# 池化 14*14*64 7*7*64
pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

# 平坦化 flat
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# 1024 神经元全连接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# dropout  50%   rate=0.5
dropout = tf.layers.dropout(dense, rate=0.5)

# 10个神经元 全连接
logits = tf.layers.dense(dropout, units=10)

# 计算误差
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# 优化器
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 精度 计算预测值和实际标签的匹配程度

# 精度。计算 预测值 和 实际标签 的匹配程度
# 返回(accuracy, update_op), 会创建两个局部变量
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),
    predictions=tf.argmax(logits, axis=1))[1]

# 创建会话
sess = tf.Session()
# 初始化变量：全局和局部
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init)

for i in range(20000):
    batch = mnist.train.next_batch(50)  # 从 Train（训练）数据集里取“下一个” 50 个样本
    train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
    if i % 100 == 0:
        test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
        print("Step={}, Train loss={}, [Test accuracy={}]".format(i, train_loss, test_accuracy))

# 测试：打印 20 个预测值 和 真实值 的对
test_output = sess.run(logits, {input_x: test_x[:20]})
inferenced_y = np.argmax(test_output, 1)
print(inferenced_y, 'Inferenced numbers')  # 推测的数字
print(np.argmax(test_y[:20], 1), 'Real numbers')  # 真实的数字

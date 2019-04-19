# -*- coding: utf-8 -*-

# @Time    : 2019-04-19 15:35
# @Author  : jian
# @File    : tf_use_save_model.py

import tensorflow as tf
v1 = tf.Variable(tf.random_normal([1,2]), name="v1")
v2 = tf.Variable(tf.random_normal([2,3]), name="v2")
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "save/model.ckpt")
    print ("V1:",sess.run(v1))
    print ("V2:",sess.run(v2))
    print ("Model restored")
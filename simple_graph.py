#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:47:36 2019

@author: aravind

This is the code for a simple computational graph in TF

"""

import numpy as np
import tensorflow as tf


c = tf.constant(5.0,name='Constant_c')

v = tf.Variable(2.0,name='Variable_v')

x = tf.placeholder(tf.float32, [1,], name='Placeholder_x')

mul = tf.multiply(v,x,name="Multiply_x_v")

y = tf.add(mul,c,name='Add_mul_c')

tf.trainable_variables()

init = tf.global_variables_initializer()


sess = tf.Session()

sess.run(init)

output = sess.run(y,feed_dict={x:[4]})
print(output)

sess.close()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    writer.close()


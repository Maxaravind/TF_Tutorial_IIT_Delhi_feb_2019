#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 00:18:49 2019

@author: aravind
"""

"""
This code let's you visualize the graph you have created in TF
"""

import tensorflow as tf

c = tf.constant(5.0,name='Constant_c')

v = tf.Variable(2.0,name='Variable_v')

x = tf.placeholder(tf.float32, [1,], name='Placeholder_x')

mul = tf.multiply(v,x,name="Multiply_x_v")

y = tf.add(mul,c,name='Add_mul_c')

init = tf.global_variables_initializer()


with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    sess.run(init)
    print(sess.run(y,feed_dict={x:[4]}))
    writer.close()
    
    
# after you run the graph, run the below command to get tensorboard running using the output folder
# tensorboard --logdir=output





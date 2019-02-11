#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:21:47 2019

@author: aravind
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:32:04 2019

@author: aravind

This is the code for making a NN with one hidden layer for mnist using TF

"""

import numpy as np
import tensorflow as tf
import keras
from pdb import set_trace as trace

tf.reset_default_graph()


num_classes = 10

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.reshape(x_train,(-1,784))
x_test = np.reshape(x_test,(-1,784))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



def get_fc_weight_variable( weight_shape,layer_name='l'):
    with tf.variable_scope(layer_name,reuse=False):
        output_channels = weight_shape[1]
        bias_shape = [output_channels]
    
        weight = tf.get_variable('Weight', weight_shape, initializer=tf.truncated_normal_initializer)
        bias   = tf.get_variable('Bias', bias_shape,   initializer=tf.truncated_normal_initializer)
    return weight, bias


#def get_fc_weight_variable( weight_shape):
#    output_channels = weight_shape[1]
#    bias_shape = [output_channels]
#
#    weight = tf.get_variable('Weight', weight_shape, initializer=tf.truncated_normal_initializer)
#    bias   = tf.get_variable('Bias', bias_shape,   initializer=tf.truncated_normal_initializer)
#    return weight, bias


batch_size = 32


x_batch = tf.placeholder(tf.float32,[batch_size,784])
y_batch = tf.placeholder(tf.float32,[batch_size,10])

fc1_w,fc1_b = get_fc_weight_variable(weight_shape=[784,20],layer_name='layer_1')

fc1 = tf.nn.sigmoid(tf.matmul(x_batch,     fc1_w) + fc1_b)

fc2_w,fc2_b = get_fc_weight_variable(weight_shape=[20,10],layer_name='layer_2')

output = tf.nn.sigmoid(tf.matmul(fc1,     fc2_w) + fc2_b)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(output - y_batch)))

optim=tf.train.GradientDescentOptimizer(0.001).minimize(loss, var_list=tf.trainable_variables())

init = tf.global_variables_initializer()

print('The Trainable variables are:\n')
for l in tf.trainable_variables():
    print(l.name)


#with tf.Session() as sess:
#    writer = tf.summary.FileWriter("output", sess.graph)
#    writer.close()

sess = tf.Session()

sess.run(init)

print('\nStaring Training.....')
for i in range(0,1000):
    indices = np.random.randint(0, x_train.shape[0], size=batch_size)
    
    feed_dict = {x_batch : x_train[indices], y_batch:y_train[indices]}

    sess.run(optim,feed_dict=feed_dict)
    
    loss_val = sess.run(loss,feed_dict=feed_dict)
    print('Loss:',loss_val)





sess.close()


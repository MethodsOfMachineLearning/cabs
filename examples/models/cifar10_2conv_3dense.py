# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 11:05:44 2016

@author: lballes
"""

import tensorflow as tf

def weight_variable(shape, stddev=1e-2):
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial)

def bias_variable(shape, val=0.05):
  initial = tf.constant(val, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def set_up_model(images, labels):
  W_conv1 = weight_variable([5, 5, 3, 64], 5e-2)
  b_conv1 = bias_variable([64], 0.0)
  h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)
  h_conv1_pool = max_pool_3x3(h_conv1)
  
  W_conv2 = weight_variable([5, 5, 64, 64], 5e-2)
  b_conv2 = bias_variable([64], 0.1)
  h_conv2 = tf.nn.relu(conv2d(h_conv1_pool, W_conv2) + b_conv2)
  h_conv2_pool = max_pool_3x3(h_conv2)
  
  batch_size = tf.gather(tf.shape(images), 0)  
  reshape = tf.reshape(h_conv2_pool, tf.pack([batch_size, -1]))
  dim = 2304
  W_fc1 = weight_variable([dim, 384], 0.04)
  b_fc1 = bias_variable([384], 0.1)
  h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
  
  W_fc2 = weight_variable([384, 192], 0.04)
  b_fc2 = bias_variable([192], 0.1)
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
  
  W_fc3 = weight_variable([192, 10], 1/192.0)
  b_fc3 = bias_variable([10], 0.0)
  h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3
  
  labels = tf.cast(labels, tf.int64)
  losses = tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc3, labels)
  return losses, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3]

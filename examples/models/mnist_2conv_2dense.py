# -*- coding: utf-8 -*-
"""
TensorFlow MNIST CNN model.
"""

import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1e-2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.05, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def set_up_model():
  tf.reset_default_graph()
  X = tf.placeholder(tf.float32, shape=[None, 784])
  y = tf.placeholder(tf.float32, shape=[None, 10])
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  X_image = tf.reshape(X, [-1,28,28,1])
  h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  h_fc2 = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
  losses = -tf.reduce_sum(y*tf.log(h_fc2), reduction_indices=[1])
  return losses, [X, y], [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]

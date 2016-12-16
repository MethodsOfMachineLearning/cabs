# -*- coding: utf-8 -*-
"""
Run CABS on a MNIST example.

This will download the dataset to data/mnist automatically if necessary.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

from cabs import CABSOptimizer

#### Specify training specifics here ##########################################
from models import mnist_2conv_2dense as model
num_steps = 8000
learning_rate = 0.1
initial_batch_size = 16
bs_min = 16
bs_max = 2048
###############################################################################

# Set up model
losses, placeholders, variables = model.set_up_model()
X, y = placeholders

# Set up CABS optimizer
opt = CABSOptimizer(learning_rate, bs_min, bs_max)
sgd_step, bs_new, loss = opt.minimize(losses, variables)

# Initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Run CABS
m = initial_batch_size
for i in range(num_steps):
  batch = mnist.train.next_batch(m)
  _, m_new, l = sess.run([sgd_step, bs_new, loss], {X: batch[0], y: batch[1]})
  print(l)
  print(m_new)
  m = m_new
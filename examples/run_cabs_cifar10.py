# -*- coding: utf-8 -*-
"""
Run CABS on a CIFAR-10 example.

This will download the dataset to data/cifar-10 automatically if necessary.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import tensorflow as tf
import cifar10_adaptive_batchsize as cifar10

from cabs import CABSOptimizer

#### Specify training specifics here ##########################################
from models import cifar10_2conv_3dense as model
num_steps = 8000
learning_rate = 0.1
initial_batch_size = 16
bs_min = 16
bs_max = 2048
###############################################################################

# Set up model
tf.reset_default_graph()
global_bs = tf.Variable(tf.constant(initial_batch_size, dtype=tf.int32))
images, labels = cifar10.inputs(eval_data=False, batch_size=global_bs)
losses, variables = model.set_up_model(images, labels)

# Set up CABS optimizer
opt = CABSOptimizer(learning_rate, bs_min, bs_max)
sgd_step, bs_new, loss = opt.minimize(losses, variables, global_bs)

# Initialize variables and start queues
sess = tf.Session()
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# Run CABS
for i in range(num_steps):
  _, m_new, l = sess.run([sgd_step, bs_new, loss])
  print(l)
  print(m_new)

# Stop queues
coord.request_stop()
coord.join(threads)
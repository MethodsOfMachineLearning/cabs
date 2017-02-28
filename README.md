# SGD with Coupled Adaptive Batch Size (CABS)

This is a TensorFlow implementation of [SGD with Coupled Adaptive Batch Size (CABS)][1].

## The Algorithm in a Nutshell

CABS is an algorithm to dynamically adapt the batch size when performing
stochastic gradient descent (SGD) on an empirical risk minimization problem. At
each iteration, it computes an empirical measure ``xi`` of the variance of the
stochastic gradient. The batch size for the next iteration is then set to
``bs_new = lr*xi/loss``, where ``lr`` is the learning rate and ``loss`` is the
current value of the loss function. Refer to the [paper][1] for more information.

## Requirements

tensorflow 1.0
## Usage

Usage of ``CABSOptimizer`` is similar to that of other TensorFlow optimizers,
with the exception that its ``minimize`` function expects a vector of ``losses``
one for each training example in the batch, instead of an aggregate mean
``loss``. This is so that the optimizer can easily access the batch size. Moreover,
some measures have to be taken to ensure that batches of appropriate size are
fed into the TensorFlow model. The specifics depend on how you choose to feed
your data.

### Manually Feeding Data
If you are a manually providing the training data via a
``feed_dict``, you have to fetch the batch size that CABS suggests and then
provide a batch of that size for the next iteration. This would look roughly
like this

```python
import tensorflow as tf
from cabs import CABSOptimizer

X, y = ... # your placeholders for data
... # set up your model
losses = ... # vector of losses, one for each training example in the batch
var_list = ... # list of trainable variables

opt = CABSOptimizer(lr, bs_min, bs_max)
sgd_step, bs_new, loss = opt.minimize(losses, var_list)
m = initial_batch_size

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(num_steps):
  X_batch, y_batch = ... # Get a batch of size m ready (you have to take care of this yourself)
  _, m_new, l = sess.run([sgd_step, bs_new, loss], feed_dict={X: X_batch, y: y_batch})
  print(l)
  print(m_new)
  m = m_new
```

The MNIST example (examples/run_cabs_mnist.py) is a full working example using
``feed_dict``.

### Reading Data from Files
If you are reading data from files using TensorFlow's built-in mechanism, the
batch size comes into action when fetching batches of data from an example queue
via ``tf.train.batch`` (or ``tf.train.shuffle_batch``). To use CABS, use a
variable ``global_bs`` as the ``batch_size`` argument of ``tf.train.batch``,
then pass ``global_bs`` to the ``minimize`` method of the ``CABSOptimizer``. The
optimizer will then write the new batch size to the global batch size variable,
directly communicating it to your data loading mechanism. Sketch:

```python
import tensorflow as tf
from cabs import CABSOptimizer

X, y = ... # your example queue
global_bs = tf.Variable(initial_batch_size) # initialize a global batch size variable
X_batch, y_batch = tf.train.batch([X, y], batch_size=global_bs) # global_bs is used as the batch_size argument of tf.train.batch
... # set up your model
losses = ... # vector of losses, one for each training example in the batch
var_list = ... # list of trainable variables

opt = CABSOptimizer(lr, bs_min, bs_max)
sgd_step, bs_new, loss = opt.minimize(losses, var_list, global_bs) # pass global_bs here, so that CABSOptimizer can write to it

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(num_steps):
  _, m_new, l = sess.run([sgd_step, bs_new, loss])
  print(l)
  print(m_new)
```

Refer to our CIFAR-10 example (examples/run_cabs_cifar10.py) for a full working
example using this mechanism.


## Quick Guide to this Implementation

The implementation of CABS (see cabs.py) itself is straight-forward. The 
``CABSOptimizer`` class inherits from ``tf.train.GradientDescentOptimizer`` and
implements the identical parameter updates, but adds the necessary additional
computations for the CABS batch size. A crucial part of that is the within-batch
estimate of the gradient variance, see equation (10) in the [paper][1]. As
mentioned in section 4.2, the computation of the second gradient moment (see
gradient_moment.py) is a little tricky; for more information see this [note][2].

[1]: https://arxiv.org/abs/1612.05086
[2]: https://drive.google.com/open?id=0B0adgqwcMJK5aDNaQ2Q4ZmhCQzA

# -*- coding: utf-8 -*-
"""
SGD optimizer with Coupled Adaptive Batch Size as described in

Lukas Balles, Javier Romero and Philipp Hennig: Coupling Adaptive Batch Sizes
with Learning Rates. [url].
"""

import tensorflow as tf
import gradient_moment as gm

class CABSOptimizer(tf.train.GradientDescentOptimizer):
  
  """Optimizer that implements stochastic gradient desent with Coupled Adative
  Batch Size (CABS) as descibed in
  
      Lukas Balles, Javier Romero and Philipp Hennig: Coupling Adaptive Batch
      Sizes with Learning Rates. [url].
  
  @@__init__
  """
  
  def __init__(self, learning_rate, bs_min=16, bs_max=2048,
               running_avg_constant=0.95, eps=0.0, c=1.0, debug=False,
               name="CABS-SGD"):
    """Construct a new gradient descent optimizer with coupled adaptive batch
    size (CABS).
    
    Args:
      :learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      :bs_min: Minimum batch size (integer). Defaults to 16.
      :bs_max: Maximum batch size (integer). Defaults to 2048.
      :running_average_constant: The variance and function value estimates
        are smoothed over iterations using an exponential running average with
        this constant. Defaults to 0.95.
      :eps: Constant added to the denominator of the CABS rule for numerical
        stability. Defaults to 0.0, but might be set to a small constant, e.g.
        eps=1e-8.
      :c: Constant by which to multiply the CABS batch size. Defaults to 1.0
        and we recommend to leave it at this.
      :debug: Boolean to switch on debug mode, where ``minimize()`` returns
        additional diagnostic outputs. Default is False.
      :name: Optional name prefix for the operations created when applying
        gradients. Defaults to "CABS-SGD".
    """
    
    super(CABSOptimizer, self).__init__(learning_rate, name=name)
    self._bs_min = bs_min
    self._bs_max = bs_max
    self._running_avg_constant = running_avg_constant
    self._eps = eps
    self._c = c
    self._debug = debug
  
  def minimize(self, losses, var_list=None, global_bs=None):
    """Add operations to minimize `loss` by updating `var_list` with SGD and
    compute the batch size for the next step according to the CABS rule.
    
    Args:
      :losses: A rank 1 `Tensor` containing the individual loss values for each
        example in the batch. You can *not* insert a scalar mean loss, as in
        other optimizers.
      :var_list: Optional list of `Variable` objects to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
      :global_bs: Optional `Variable` to which the computed batch size is
        assigned. When you feed data using tensorflow queues, use this variable
        as batch size in ``tf.train.batch()`` or `tf.train.shuffle_batch`. When
        you feed data via ``placeholder``s and ``feed_dict``s, use
        ``global_bs=None``. In this case you have to fetch ``bs_new_int``
        (one of the return values of this function, see below) and take care
        of the batch size yourself.
    Returns:
      If ``debug=False``
        :sgd_step: An Operation that updates the variables in `var_list` via
          SGD step.
        :bs_new: A scalar integer tensor containing the CABS batch size for the
          next optimization step.
        :loss: A scalar tensor with the mean of the inserted ``losses``.
      If ``debug=True``
        :sgd_step: An Operation that updates the variables in `var_list` via
          SGD step.
        :bs_new: A scalar integer tensor containing the rounded and capped CABS
          batch size to be used in the next optimization step.
        :bs_new_raw: A scalar tensor containing the raw CABS batch size before
          rounding and capping.
        :loss_avg: A scalar tensor containing the running average of the mean
          loss.
        :loss: A scalar tensor with the mean of the inserted ``losses``, i.e.
          the current loss.
        :xi_avg: A scalar tensor containing the running average of the
          gradient variance.
        :xi: A scalar tensor containing the current gradient variance.
      If ``global_bs`` was not ``None``, the result ``bs_new`` is also
      written to the ``global_bs`` Variable.
    Raises:
      ValueError: If some of the variables are not `Variable` objects.
    """
    
    if global_bs is not None:
      assert isinstance(global_bs, tf.Variable)
    
    # Create variables for the moving averages of noise level and loss
    if var_list is None:
      var_list = tf.trainable_variables()
    xi_avg = tf.Variable(0.0)
    loss_avg = tf.Variable(1.0)
    
    # Extract input data type and batch size from the provided losses
    input_dtype = losses.dtype.base_dtype
    input_batch_size = tf.cast(tf.gather(tf.shape(losses), 0), input_dtype)
    
    # Convert constant algo parameters to tensors
    mu = tf.convert_to_tensor(self._running_avg_constant, dtype=input_dtype)
    c = tf.convert_to_tensor(self._c, dtype=input_dtype)
    lr = tf.convert_to_tensor(self._learning_rate, dtype=input_dtype)
    eps = tf.convert_to_tensor(self._eps, dtype=input_dtype)
    bs_min = tf.convert_to_tensor(self._bs_min, dtype=input_dtype)
    bs_max = tf.convert_to_tensor(self._bs_max, dtype=input_dtype)
        
    # Compute mean loss and feed it into a running average
    loss = tf.reduce_mean(losses)
    update_avgs = [loss_avg.assign(mu*loss_avg + (1.0-mu)*loss)]
    
    # Compute gradients and gradient moments
    grads, moms = gm.grads_and_grad_moms(loss, input_batch_size, var_list)
    grads_squared = [tf.square(g) for g in grads]
    
    # Compute gradient variance and feed it into a running average
    grad_variances = [(m-g2) for g2, m in zip(grads_squared, moms)]
    xi = tf.add_n([tf.reduce_sum(gv) for gv in grad_variances])
    update_avgs.append(xi_avg.assign(mu*xi_avg + (1.0-mu)*xi))
    
    # Compute the new batch size (with a dependency that makes sure that the
    # moving averages are updated beforehand)
    with tf.control_dependencies(update_avgs):
      bs_new_raw = c*lr*tf.divide(xi_avg, loss_avg+eps)
    
    # Round the new batch size
    bs_new_rounded = tf.round(bs_new_raw)
    bs_new = tf.clip_by_value(bs_new_rounded, bs_min, bs_max)
    bs_new = tf.to_int32(bs_new)
        
    # If a global variable to hold the batch size was given by the user, add
    # operation that saves the new batch size to this variable
    deps = [bs_new]
    if global_bs is not None:
      deps.append(global_bs.assign(bs_new))
    
    # Add SGD update operations
    with tf.control_dependencies(deps):
      sgd_updates = [v.assign_sub(lr*g) for v, g in zip(var_list, grads)]
      sgd_step = tf.group(*sgd_updates)
      
    # Return the SGD update op and the new (rounded) batch size
    # In debug mode, additionally return the various intermediate quantities
    if self._debug:
      return sgd_step, bs_new, bs_new_raw, loss_avg, loss, xi_avg, xi
    else:
      return sgd_step, bs_new, loss
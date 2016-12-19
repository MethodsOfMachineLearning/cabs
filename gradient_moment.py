# -*- coding: utf-8 -*-
"""
Computation of *moments* of gradients through tensorflow operations.

Tensorflow is typically used for empircal risk minimzation with gradient-based
optimization methods. That is, we want to adjust trainable variables ``W``,
such as to minimize an objective quantity, called ``LOSS``, of the form

    LOSS(W) = (1/n) * sum{i=1:n}[ loss(W, d_i) ]

That is the mean of individual losses induced by ``n`` training data points
``d_i``. Consquently, the gradient of ``LOSS`` w.r.t. the variables ``W`` is
the mean of individual gradients ``dloss(W, d_i)``. These individual gradients
are not computed separately when we call ``tf.gradients`` on the aggregate
``LOSS``. Instead, they are implicitly aggregated by the operations in the
backward graph. This batch processing is crucial for the computational
efficiency of the gradient computation.

This module provides functionality to compute the ``p``-th moment of the
individual gradients, i.e. the quantity

    MOM(W) = (1/n) * sum{i=1:n}[ dloss(w, d_i)**p ]

without giving up the efficiency of batch processing. For a more detailed 
explanation, see the note [1]. Applications of this are the computation of the 
gradient variance estimate in [2] and [3].

[1] https://drive.google.com/open?id=0B0adgqwcMJK5aDNaQ2Q4ZmhCQzA

[2] M. Mahsereci and P. Hennig. Probabilistic line searches for stochastic
optimization. In Advances in Neural Information Processing Systems 28, pages
181-189, 2015.

[3] L. Balles, J. Romero and P. Hennig. Coupling Adaptive Batch Sizes with
Learning Rates. In arXiv preprint arXiv:1612.05086, 2016.
https://arxiv.org/abs/1612.05086.
"""

import tensorflow as tf
from tensorflow.python.ops import gen_array_ops

VALID_TYPES = ["MatMul", "Conv2D", "Add"]
VALID_REGULARIZATION_TYPES = ["L2Loss"]

def _check_and_sort_ops(op_list):
  """Sort a list of ops according to type into valid types for which we can
  compute the gradient moment) and regularizers. Raise an exception when
  encountering an op of invalid type."""
  
  valid, regularizers = [], []
  for op in op_list:
    if op.type in VALID_TYPES:
      valid.append(op)
    elif op.type in VALID_REGULARIZATION_TYPES:
      regularizers.append(op)
    else:
      raise Exception("A variable in var_list is consumed by an operation of "
          "type {} for which I don't how to compute the gradient moment. "
          "Allowed are types {} and regularization operations "
          "of type {}".format(op.type, str(VALID_TYPES),
          str(VALID_REGULARIZATION_TYPES)))
  return valid, regularizers

def grads_and_grad_moms(loss, batch_size, var_list, mom=2):
  """Compute the gradients and gradient moments of ``loss`` w.r.t. to the
  variables in ``var_list``
  
  Inputs:
      :loss: The tensor containing the scalar loss. The loss has  to be the
          ``tf.mean`` of ``batch_size`` individual losses induced by
          individual training data points.
      :batch_size: Self-explanatory. Integer tensor.
      :var_list: The list of variables.
      :mom: The desired moment. Integer. Defaults to 2.
  
  Returns:
      :v_grads: The gradients of ``loss`` w.r.t. the variables in ``var_list``
          as computed by ``tf.gradients(loss, var_list)``.
      :grad_moms: The gradient moments for each variable in ``var_list``."""
      
  assert len(set(var_list)) == len(var_list)           
  vs = [tf.convert_to_tensor(v) for v in var_list]
  num_vars = len(vs)
  
  consumers = []
  consumer_outs = []
  for v in vs:
    valid, regularizers = _check_and_sort_ops(v.consumers())
    if len(valid) > 1:
      raise Exception("Variable {} is consumed by more than one operation "
          "(ignoring regularization operations)".format(v.name))
    if len(regularizers) > 1:
      raise Exception("Variable {} is consumed by more than one "
          "regularization operation".format(v.name))
    consumers.extend(valid)
    consumer_outs.extend(valid[0].outputs)      
  
  # Use tf.gradients to compute gradients w.r.t. the variables, while also
  # retrieving gradients w.r.t. the outputs
  all_grads = tf.gradients(loss, vs+consumer_outs)
  v_grads = all_grads[0:num_vars]
  out_grads = all_grads[num_vars::]
  
  # Compute the gradient moment for each (v, vp, op, output)
  with tf.name_scope("grad_moms"):
    grad_moms = [_GradMom(o, v, out_grad, batch_size, mom)
                for o, v, out_grad in zip(consumers, vs, out_grads)]
  
  return (v_grads, grad_moms)

def _GradMom(op, v, out_grad, batch_size, mom=2):
  """Wrapper function for the operation type-specific GradMom functions below.
  
  Inputs:
      :op: A tensorflow operation of type in VALID_TYPES.
      :v: The read-tensor of the trainable variable consumed by this operation.
      :out_grad: The tensor containing the gradient w.r.t. to the output of
          the op (as computed by ``tf.gradients``).
      :batch_size: Batch size ``m`` (constant integer or scalar int tf.Tensor)
      :mom: Integer moment desired (defaults to 2)."""
  
  with tf.name_scope(op.name+"_grad_mom"):
    if op.type == "MatMul":
      return _MatMulGradMom(op, v, out_grad, batch_size, mom)
    elif op.type == "Conv2D":
      return _Conv2DGradMom(op, v, out_grad, batch_size, mom)
    elif op.type == "Add":
      return _AddGradMom(op, v, out_grad, batch_size, mom)
    else:
      raise ValueError("Don't know how to compute gradient moment for "
          "variable {}, consumed by operation of type {}".format(v.name,
          op.type))

def _MatMulGradMom(op, W, out_grad, batch_size, mom=2):
  """Computes gradient moment for a weight matrix through a MatMul operation.
  
  Assumes ``Z=tf.matmul(A, W)``, where ``W`` is a d1xd2 weight matrix, ``A``
  are the nxd1 activations of the previous layer (n being the batch size).
  ``out_grad`` is the gradient w.r.t. ``Z``, as computed by ``tf.gradients()``.
  No transposes in the MatMul operation allowed.
  
  Inputs:
      :op: The MatMul operation
      :W: The weight matrix (the tensor, not the variable)
      :out_grad: The tensor of gradient w.r.t. to the output of the op
      :batch_size: Batch size n (constant integer or scalar int tf.Tensor)
      :mom: Integer moment desired (defaults to 2)"""
  
  assert op.type == "MatMul"
  t_a, t_b = op.get_attr("transpose_a"), op.get_attr("transpose_b")
  assert W is op.inputs[1] and not t_a and not t_b
  
  A = op.inputs[0]
  out_grad_pow = tf.pow(out_grad, mom)
  A_pow = tf.pow(A, mom)
  return tf.mul(batch_size, tf.matmul(A_pow, out_grad_pow, transpose_a=True))

def _Conv2DGradMom(op, f, out_grad, batch_size, mom=2):
  """Computes gradient moment for the filter of a Conv2D operation.
  
  Assumes ``Z=tf.nn.conv2d(A, f)``, where ``f`` is a ``[h_f, w_f, c_in, c_out]``
  convolution filter and ``A`` are the ``[n, h_in, w_in, c_in]`` activations of
  the previous layer (``n`` being the batch size). ``out_grad`` is the gradient
  w.r.t. ``Z``, as computed by ``tf.gradients()``.
  
  Inputs:
      :op: The Conv2D operation
      :f: The filter (the tensor, not the variable)
      :out_grad: The tensor of gradient w.r.t. to the output of the op
      :batch_size: Batch size ``n`` (constant integer or scalar int tf.Tensor)
      :mom: Integer moment desired (defaults to 2)"""
  
  assert op.type == "Conv2D"
  assert f is op.inputs[1]
  
  strides = op.get_attr("strides")
  padding = op.get_attr("padding")
  use_cudnn = op.get_attr("use_cudnn_on_gpu")
  data_format = op.get_attr("data_format")
  
  inp = op.inputs[0]
  inp_pow = tf.pow(inp, mom)
  
  f_shape = tf.shape(f)
  out_grad_pow = tf.pow(out_grad, mom)
  
  raw_moment = tf.nn.conv2d_backprop_filter(inp_pow, f_shape, out_grad_pow,
      strides, padding, use_cudnn, data_format)
  return tf.mul(batch_size, raw_moment)

def _AddGradMom(op, b, out_grad, batch_size, mom=2):
  """Computes gradient moment for a bias variable through an Add operation.
  
  Assumes ``Z = tf.add(Zz, b)``, where ``b`` is a bias parameter and ``Zz`` is
  a ``[n, ?]`` tensor (``n`` being the batch size). Broadcasting for all kinds
  of shapes of ``Zz`` (e.g. ``[n, d_in]`` or ``[n, h_in, w_in, c_in]`` are
  supported. ``out_grad`` is the gradient w.r.t. ``Z``, as computed by
  ``tf.gradients()``.
  
  Inputs:
      :op: The Add operation
      :b: The bias parameter (the tensor, not the variable)
      :out_grad: The tensor of gradient w.r.t. to the output of the op
      :batch_size: Batch size ``n`` (constant integer or scalar int tf.Tensor)
      :mom: Integer moment desired (defaults to 2)"""
  
  assert op.type == "Add"
  
  out_grad_pow = tf.pow(out_grad, mom)
  
  if b is op.inputs[0]:
    y = op.inputs[1]
    sx = tf.shape(b)
    sy = tf.shape(y)
    rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
    raw_mom = tf.reshape(tf.reduce_sum(out_grad_pow, rx), sx)
  elif b is op.inputs[1]:
    x = op.inputs[0]
    sx = tf.shape(x)
    sy = tf.shape(b)
    rx, ry = gen_array_ops._broadcast_gradient_args(sx, sy)
    raw_mom = tf.reshape(tf.reduce_sum(out_grad_pow, ry), sy)
  return tf.mul(batch_size, raw_mom)

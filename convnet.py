import numpy as np

from layers import *
from fast_layers import *
from layer_utils import *


def two_layer_convnet(X, model, y=None, reg=0.0):
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape

  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads


def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model


pass
def three_layer_convnet(X, model, y=None, reg=0.0):
  
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  W3, b3= model['W3'], model['b3']
  N, C, H, W = X.shape

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  a2, cache2 = conv_relu_forward(a1, W2, b2, conv_param, pool_param)
  scores, cache3 = affine_forward(a2, W3, b3)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW3, db3 = affine_backward(dscores, cache3)
  da2,  dW2, db2 = conv_relu_backward(da1, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da2, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
  


  return loss, grads



def init_three_layer_convnet(weight_scale=1e-3, bias_scale=0,
                 input_shape=(3, 32, 32), num_classes=10, num_filters=32, filter_size=5):
  
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  # First six convolutional-layer
  # architecture =[conv-relu-pooling]x2--> conv-relu --> affine 

  if num_filters == 32:
    num_filters=np.ones((1,5))*32

  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['W3'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  model['b3'] = bias_scale * np.random.randn(num_classes)
  

  return model



def init_five_layer_convnet(weight_scale=1e-3, bias_scale=0,
                 input_shape=(3, 32, 32), num_classes=10, num_filters=32, filter_size=5):
  
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  # First six convolutional-layer
  # architecture =[conv-relu-pooling]x2--> conv-relu --> affine 

  if num_filters == 32:
    num_filters=np.ones((1,5))*32

  #layer-1 --> conv-relu-pooling
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b2'] = bias_scale * np.random.randn(num_filters)
  model['W3'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b3'] = bias_scale * np.random.randn(num_filters)
  model['W4'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b4'] = bias_scale * np.random.randn(num_filters)
  model['W5'] = weight_scale * np.random.randn(num_filters * H * W / 8, num_classes)
  model['b5'] = bias_scale * np.random.randn(num_classes)
  return model


def five_layer_convnet(X, model, y=None, reg=0.0):
  
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  W3, b3, W4, b4,W5, b5= model['W3'], model['b3'], model['W4'], model['b4'],model['W5'], model['b5']
  N, C, H, W = X.shape

  
  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
  a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv_param, pool_param)
  a3, cache3 = conv_relu_pool_forward(a2, W3, b3, conv_param, pool_param)
  a4, cache4 = conv_relu_forward(a3, W4, b4, conv_param, pool_param)
  scores, cache5 = affine_forward(a4, W5, b5)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW3, db3 = affine_backward(dscores, cache5)
  da2,  dW2, db2 = conv_relu_backward(da1, cache4)
  da3,  dW3, db3 = conv_relu_pool_backward(da2, cache3)
  da4,  dW2, db2 = conv_relu_pool_backward(da3, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da4, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  dW5 += reg * W5
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3,W4, W5])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3, 'W4': dW4, 'b4': db4, 'W5': dW5, 'b5': db5}  



  return loss, grads

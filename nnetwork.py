import numpy as np
from layers import *




def init_three_layer_neuralnet(weight_scale=1, bias_scale=0, input_feat_dim=786,num_classes=10, num_neurons=(20, 30)):

  
  assert len(num_neurons)  == 2, 'You must provide number of neurons for two layers...'

  model = {}
  model['W1'] = (np.random.randn(input_feat_dim,num_neurons[0]) * weight_scale) * np.sqrt(2.0/input_feat_dim) # Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  model['b1'] = np.zeros(num_neurons[0])# Initialize with zeros
  
  model['W2'] = (np.random.randn(num_neurons[0],num_neurons[1]) * weight_scale)* np.sqrt(2.0/num_neurons[0]) # Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  model['b2'] = np.zeros(num_neurons[1]) # Initialize with zeros

  model['W3'] = (np.random.randn(num_neurons[1],num_classes) * weight_scale)* np.sqrt(2.0/num_neurons[1])# Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  model['b3'] = np.zeros(num_classes) # Initialize with zeros

  return model



def three_layer_neuralnetwork(X, model, y=None, reg=0.0,verbose=0):
  
  # Unpack weights
  W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3']
  N,D= X.shape

  assert W1.shape[0] == D, ' W1 2nd dimenions must match number of features'
  
  dW1,dW2,dW3,db1,db2,db3=np.zeros_like(W1),np.zeros_like(W2),np.zeros_like(W3),np.zeros_like(b1),np.zeros_like(b2),np.zeros_like(b3)
  # Compute the forward pass
  relu=lambda x:np.maximum(x,0)
  a1= relu(np.dot(X,W1)+b1)
  a2= relu(np.dot(a1,W2)+b2)
  a3 =(np.dot(a2,W3)+b3)
  scores=a3
  probs = np.exp(a3 - np.max(a3, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = a3.shape[0]
  data_loss = -np.sum(np.log(probs[np.arange(N), y])) / N

  
  if y is None:
    return scores

  # Compute the backward pass
  z3=np.exp(np.dot(a2,W3)+b3)

  #derivative of affine-softmax layer
  ones= np.zeros_like(scores)
  ones[ range(scores.shape[0]), y] = 1;
  delta3 = - 1.0/N * ( ones -  ( z3 / z3.sum(axis=1,keepdims=True) ) )
  dW3=np.dot(delta3.T,a2).T
  db3=np.sum(delta3,axis=0)

  #derivative of affine-Relu
  #Layer-2
  delta2=np.dot(delta3,W3.T) * (a2 > 0)
  dW2=np.dot(delta2.T,a1).T
  db2=np.sum(delta2,axis=0)
  #Layer -1
  delta1=np.dot(delta2,W2.T) * (a1 > 0)
  dW1=np.dot(delta1.T,X).T
  db1=np.sum(delta1,axis=0)
  #
  reg_loss =  reg * ( (W1 ** 2).sum() + (W2 ** 2).sum() + (W3 ** 2).sum() )

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,'W3':dW3,'b3':db3}
  
  return loss, grads


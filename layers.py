import numpy as np

def affine_forward(x, w, b):

  out = None

  x_shape = x.shape
  x = x.reshape( (x.shape[0], -1) )
  print (x.shape,x_shape,w.shape)
  out = np.dot(x, w) + b
  x = x.reshape(x_shape)

  
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):

  x, w, b = cache
  dx, dw, db = None, None, None
 
  N = x.shape[0]
  D = w.shape[0]
  M = w.shape[1]

  x_shape =  x.shape
  x = x.reshape(x.shape[0], -1)
  dw = np.dot(x.T, dout)
  dx = np.dot(dout, w.T)
  db = dout.sum(axis=0)

  dx = dx.reshape(x_shape)

  return dx, dw, db


def relu_forward(x):

  out = None

  relu=lambda x:np.maximum(x,0)
  out=relu(x)

  
  cache = x
  return out, cache


def relu_backward(dout, cache):

  dx, x = None, cache
 
  dx = dout * (x > 0)
  

  return dx


def svm_loss(x, y):

  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


def im2col(A, B, skip=(1,1)):

  D,M,N = A.shape
  col_extent = N - B[1] + 1
  row_extent = M - B[0] + 1


  # Get Starting block indices
  start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

  # Get Depth indeces
  cidx=M*N*np.arange(D)
  start_idx=(cidx[:,None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

  # Get offsetted indices across the height and width of input array
  offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

  # Get all actual indices & index into input array for final output
  out = np.take (A,start_idx.ravel()[:,None] + offset_idx[::skip[0],::skip[1]].ravel())
  return out

def im2colidx(A, B, skip=(1,1)):

  D,M,N = A.shape
  col_extent = N - B[1] + 1
  row_extent = M - B[0] + 1


  # Get Starting block indices
  start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

  # Get Depth indeces
  cidx=M*N*np.arange(D)
  start_idx=(cidx[:,None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

  # Get offsetted indices across the height and width of input array
  offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

  # Get all actual indices & index into input array for final output
  out_idx=start_idx.ravel()[:,None] + offset_idx[::skip[0],::skip[1]].ravel()
  
  return out_idx


def conv_forward_naive(x, w, b, conv_param):


  out = None
  
  stride = conv_param['stride']
  pad = conv_param['pad']

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  H_out = 1 + (H+2 * pad - HH) / stride
  W_out = 1 + (W+2 * pad - WW) / stride
  out = np.zeros((N, F, H_out, W_out))
  img_pad = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  for i in range(N):
    Img_col=im2col(img_pad[i,:,:,:],(HH,WW),(stride,stride))
    feature_map=w.reshape((F,-1)).dot(Img_col) + b.reshape(-1,1)
    out[i,:,:,:]=feature_map.reshape((F,H_out,W_out))
  
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):

  dx, dw, db = None, None, None
  
  x, w, b, conv_param = cache
  stride = conv_param['stride']
  pad = conv_param['pad']

  db = np.sum(dout, axis=(0, 2, 3))

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H_out = 1 + (H+2 * pad - HH) / stride
  W_out = 1 + (W+2 * pad - WW) / stride
  Img_pad = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  dw=np.zeros_like(w)
  '''for i in range(N):
    for k in range(F):
      Img_col=im2col(Img_pad[i,:,:,:],(HH,WW),(stride,stride))
      dout_reshaped=np.ravel( dout[i, k] )
      dout_tile=np.tile(dout_reshaped,(C*WW*HH,1))
      print (dout_reshaped.shape,dout_tile.shape)
      dw[i,:,:,:]=np.sum(Img_col*dout_tile,axis=1).reshape(C,HH,WW)'''


  dx = np.zeros_like(x)
  dx_pad = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  
  for n in range(N):
    for f in range(F):
      for i in range(int(H_out)):
        for j in range(int(W_out)):
          Img_window = Img_pad[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW]

          dw[f] += Img_window * dout[n, f, i, j]

          dx_pad[n, :, i * stride : i * stride + HH, j * stride : j * stride + WW] += w[f] * dout[n, f, i, j]
  
  dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  
  out = None

  N, C, H, W = x.shape
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = (H - HH) / stride + 1
  W_out = (W - WW) / stride + 1

  out = np.zeros([N, C, H_out, W_out])
  for n in range(N):
    for c in range(C):
      for i in range(int(H_out)):
        for j in range(int(W_out)):
          window = x[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW]
          out[n, c, i, j] = np.max(window)
 
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):

  dx = None
 

  x, pool_param = cache
  N, C, H, W = x.shape
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']

  H_out = (H - HH) / stride + 1
  W_out = (W - WW) / stride + 1

  dx = np.zeros_like(x)
  for n in range(N):
    for c in range(C):
      for i in range(int(H_out)):
        for j in range(int(W_out)):
          window = x[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW]
          x_max = np.max(window)
          dx[n, c, i * stride : i * stride + HH, j * stride : j * stride + WW] = (window == x_max) * dout[n, c, i, j]
 
  return dx


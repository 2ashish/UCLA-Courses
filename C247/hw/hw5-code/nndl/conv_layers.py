import numpy as np
from nndl.layers import *
import pdb


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N,C,H,W = x.shape
  F, C, HH, WW = w.shape
  npad = ((0, 0), (0, 0), (pad, pad), (pad, pad))
  xpad = np.pad(x,npad)
  
  H_ = int(1 + (H + 2 * pad - HH) / stride)
  W_ = int(1 + (W + 2 * pad - WW) / stride)
  out = np.zeros((N,F,H_,W_))
  for i in range(N):
    for j in range(F):
        for h in range(H_):
            for width in range(W_):
                cur=0
                for i2 in range(C):
                    for j2 in range(HH):
                        for k2 in range(WW):
                            cur+=xpad[i,i2,h*stride+j2,width*stride+k2]*w[j,i2,j2,k2]
                out[i,j,h,width] = cur + b[j]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  db=np.sum(dout,axis=(0,2,3))
  dxpad=np.zeros(xpad.shape)
  dw=np.zeros(w.shape)

    
  N,C,H,W = x.shape
  F, C, HH, WW = w.shape
  H_ = int(1 + (H + 2 * pad - HH) / stride)
  W_ = int(1 + (W + 2 * pad - WW) / stride)
  for i in range(N):
    for j in range(F):
        for h in range(H_):
            for width in range(W_):
                cur=0
                for i2 in range(C):
                    for j2 in range(HH):
                        for k2 in range(WW):
                            #cur+=xpad[i,i2,h*stride+j2,width*stride+k2]*w[j,i2,j2,k2]
                            dxpad[i,i2,h*stride+j2,width*stride+k2] += w[j,i2,j2,k2]*dout[i,j,h,width]
                            dw[j,i2,j2,k2] += xpad[i,i2,h*stride+j2,width*stride+k2]*dout[i,j,h,width]
                #out[i,j,h,width] = cur + b[j]
  dx = dxpad[:,:,pad:-pad, pad:-pad]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  s = pool_param['stride']
  H_ = int(1 + (H-ph)/s)
  W_ = int(1 + (W-pw)/s)
  out = np.zeros((N,C,H_,W_))
  for i in range(N):
    for j in range(C):
        for h in range(H_):
            for w in range(W_):
                out[i,j,h,w]=np.max(x[i,j,h*s:h*s+ph,w*s:w*s+pw])
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  dx = np.zeros(x.shape)
  N, C, H, W = x.shape
  ph = pool_param['pool_height']
  pw = pool_param['pool_width']
  s = pool_param['stride']
  H_ = int(1 + (H-ph)/s)
  W_ = int(1 + (W-pw)/s)
  for i in range(N):
    for j in range(C):
        for h in range(H_):
            for w in range(W_):
                cur=np.max(x[i,j,h*s:h*s+ph,w*s:w*s+pw])
                for i2 in range(ph):
                    for j2 in range(pw):
                        if x[i,j,h*s+i2,w*s+j2]==cur:
                            dx[i,j,h*s+i2,w*s+j2]=dout[i,j,h,w]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  xb = np.transpose(x, (0,2,3,1))
  xb = xb.reshape((-1,C))
  out,cache = batchnorm_forward(xb, gamma, beta, bn_param)
  out = out.reshape((N,H,W,C))
  out = np.transpose(out, (0,3,1,2))
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = dout.shape
  dout = np.transpose(dout, (0,2,3,1))
  dout = dout.reshape(-1,C)
  #x,x_n,mean,var,eps,gamma = cache
  #x = x.reshape((N,H,W,C))
  #x = np.transpose(x, (0,3,1,2))
  #x_n = x_n.reshape((N,H,W,C))
  #x_n = np.transpose(x_n, (0,3,1,2))
  #cache = x,x_n,mean,var,eps,gamma
  dx, dgamma, dbeta = batchnorm_backward(dout,cache)
  dx = dx.reshape((N,H,W,C))
  dx = np.transpose(dx, (0,3,1,2))
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta
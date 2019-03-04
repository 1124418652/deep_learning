#-*- coding:utf-8 -*-
from __future__ import division
import os
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import signal


def zero_pad(X, pad, value = 0):
	"""
	Pad with zeros all images of the dataset X. The padding is applied to the
	height and width of an image

	Argument:
	X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch
	of m images
	pad -- integer, amount of padding around each image on vertical and 
	horizontal dimensions
	value -- integer, the value of pad

	Returns:
	X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
	"""

	X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
		'constant', constant_values = value)
	return X_pad

def conv_single_step(a_slice_prev, W, b):
	"""
	Apply one filter defined by parameters W on a single slice (a_slice_prev) of
	the output activation of the previous layer.

	Arguments:
	a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
	W -- Weight parameters contained in window - matrix of shape (f, f, n_C_prev)
	b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

	Returns:
	Z -- a scalar value, result of convolving the sliding window (W, b) on a slice 
	x of the input data
	"""

	s = a_slice_prev * W + b 
	Z = np.sum(s)

	return Z

def conv_forward(A_prev, W, b, hparameters):
	"""
	Implements the forward propagation for a convolution function

	Arguments:
	A_prev -- output activations of the previous layer, numpy array of shape
	(m, n_H_prev, n_W_prev, n_C_prev)
	W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
	b -- Biases, numpy array of shape (1, 1, 1, n_C)
	hparameters -- Python dictionary containing "stride" and "pad"

	Returns:
	Z -- convolutional output, numpy array of shape (m, n_H, n_W, n_C)
	cache -- cache of values needed for the conv_backward() function, contains 
	the input of this function
	"""

	if not A_prev.ndim == W.ndim == 4:
		raise ValueError('dimensions of A_prev and W must be 4')

	assert(A_prev.shape[3] == W.shape[2])

	# preparation parameters
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	(f, f, _, n_C) = W.shape

	stride = hparameters['stride']
	pad  = hparameters['pad']

	n_H = (n_H_prev + 2 * pad - f) // stride + 1
	n_W = (n_W_prev + 2 * pad - f) // stride + 1

	Z = np.zeros((m, n_H, n_W, n_C))        # the array after convolve
	A_prev_pad = zero_pad(A_prev, pad)      # pad the previous array

	a_prev_pad = A_prev_pad[i]    # select i'th training example	
	for i in range(m):                # loop over the batch of training examples
		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):

					vert_start = h * stride 
					vert_end = vert_start + f 
					horiz_start = w * stride 
					horiz_end = horiz_start + f 

					a_slice_prev = a_prev_pad[vert_start: vert_end, 
						horiz_start: horiz_end, :]
					Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], 
						b[:, :, :, c])

	cache = (A_prev, W, b, hparameters)
	return Z, cache 

def pool_forward(A_prev, hparameters, mode = 'max'):
	"""
	Implements the forward pass of the pooling layer

	Arguments:
	A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	hparameters -- python dictionary containing 'f' and 'stride'
	mode -- the pooling mode you would like to use, defined as a string 
	('max' or 'average')

	Returns:
	A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
	cache -- cache used in the backward pass of the pooling layer, contains the
	input and hparameters
	"""

	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	f = hparameters['f']
	stride = hparameters['stride']

	n_H = (n_H_prev - f) // stride + 1
	n_W = (n_W_prev - f) // stride + 1
	n_C = n_C_prev

	A = np.zeros((m, n_H, n_W, n_C))

	for h in range(n_H):
		for w in range(n_W):
			for c in range(n_C):
				vert_start = h * stride
				vert_end = vert_start + f 
				horiz_start = w * stride
				horiz_end = horiz_start + f 

				a_prev_slice = A_prev[:, vert_start: vert_end, horiz_start: horiz_end, c]

				if 'max' == mode:
					A[:, h, w, c] = np.max(a_prev_slice, axis = (1, 2))
				elif 'average' == mode:
					A[:, h, w, c] = np.mean(a_prev_slice, axis = (1, 2))

	cache = (A_prev, f, stride)
	return A, cache

def pool_backward(dA, cache, mode = 'max'):
	"""
	Implements the backward pass of the pooling layer

	Arguments:
	dA -- gradient of cost with respect to the output of the pooling layer,
	same shape as A
	cache -- cache output from the forward pass of the pooling layer, contains
	the layer's input and hparameters which contains pooling's f and stride

	Returns:
	dA_prev -- gradient of cost with respect to the input of the pooling layer, same 
	shape as A_prev
	"""

	(A_prev, hparameters) = cache
	stride = hparameters['stride']
	f = hparameters['f']

	m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
	m, n_H, n_W, n_C = dA.shape

	def get_mask(a_slice, mode = 'max'):
		"""
		Create a mask from an input matrix x

		Arguments:
		a_slice -- Array of shape (f, f)
		mode -- the mode of pooling layer, 'max' or 'average'

		Returns:
		mask -- Array of the same shape as window, contains a True at the position corresponding
		to the max entry of x, or every element if 1/(f*f) if the mode is average 
		"""

		if not 2 == a_slice.ndim:
			raise ValueError("the dimension of array a_slice must be 2") 

		if 'max' == mode:
			mask = (a_slice == np.max(a_slice))

		elif 'average' == mode:
			mask = np.ones(a_slice.shape) / (a_slice.shape[0] * a_slice.shape[1])

		return mask 

	dA_prev = np.zeros(A_prev.shape)

	for i in range(m):
		a_prev = A_prev[i]
		for h in range(n_H):
			for w in range(n_W):
				for c in range(n_C):
					vert_start = h * stride
					vert_end = vert_start + f 
					horiz_start = w * stride
					horiz_end = horiz_start + f 

					a_prev_slice = a_prev[vert_start: vert_end, horiz_start: horiz_end, c]
					mask = get_mask(a_prev_slice, mode)
					dA_prev[i, vert_start: vert_end, horiz_start, horiz_end, c] += mask *\
						dA[i, n_H, n_W, c]
	return dA_prev

def conv_backward(dZ, cache):
	"""
	Implement the backward propagation for a convolution function
	dA_prev will be calculated as:
	convolve2d(dZ[i, :, :, c], w[:, :, c_prev, c], mode = 'full', boundary = 'fill', fillvalue = 0)
	dW will be calculated as:
	convolve2d(A_prev[i, :, :, c_prev], dZ[i, :, :, ])

	Arguments:
	dZ -- gradient of the cost with respect to the output of the conv layer(Z), numpy 
	array of shape (m, n_H, n_W, n_C)
	cache -- cache of values needed for the conv_backward(), output of conv_forward()

	Returns:
	dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
	numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
	dW -- gradient of the cost with respect to the weight of the conv layer (W), numpy 
	array of shape (f, f, n_C_prev, n_C)
	db -- gradient of the cost with respect to the biases of the conv layer (b), numpy 
	array of shape (1, 1, 1, n_C)
	"""

	(A_prev, W, b, hparameters) = cache
	(m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
	(f, f, n_C_prev, n_C) = W.shape

	stride = hparameters['stride']
	pad = hparameters['pad']

	(m, n_H, n_W, n_C) = dZ.shape
	dA_prev = np.zeros(A_prev.shape)
	dW = np.zeros(W.shape)
	db = np.zeros(b.shape)

	A_prev_pad = zero_pad(A_prev, pad)
	dA_prev_pad = np.zeros(A_prev_pad.shape)


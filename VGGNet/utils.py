#-*- coding: utf-8 -*-
"""
Created on 2019/5/9

@Author xhj
"""

import numpy as np 
import tensorflow as tf


def conv(layer_name, X, kernel_num, kernel_size = (3, 3), 
		 strides = [1, 1, 1, 1], is_pretrain = True):
	"""
	执行前向的卷积运算的单元

	Args:
		layer_name: str 类型，表示该层卷积的名字
		X: 输入 tensor，维数为 [batch_size, height, width, channels]
		kernel_num: int 类型，表示卷积核的数目
		kernel_size: list 或 tupple 类型，表示卷积核的大小，(height, width)
		strides: list 类型，表示卷积的步长
		is_pretrain: boolean 类型，表示该层的卷积核参数是否需要更新

	Returns:
		X: 卷积运算之后的输出，由于卷积过程中的padding都是使用'SAME'的，所以
		   维数和输出张量一样
	"""

	in_channels = X.shape[-1]
	with tf.variable_scope(layer_name):      # 使用 tf.variable_scope(name) 会在内部的变量名之前加上 'name/'
		W = tf.get_variable(name = 'weights',
							shape = [kernel_size[0], kernel_size[1], in_channels, kernel_num],
							dtype = tf.float32,
							initializer = tf.contrib.layers.xavier_initializer(),
							trainable = is_pretrain)
		b = tf.get_variable(name = 'biases',
							shape = [kernel_num],
							dtype = tf.float32,
							initializer = tf.constant_initializer(0.0),
							trainable = is_pretrain)
		X = tf.nn.conv2d(X, W, strides = strides, padding = 'SAME', name = 'conv')
		X = tf.nn.bias_add(X, b, name = 'bias_add')

		return X

def pool(layer_name, X, ksize = [1, 2, 2, 1], 
		 strides = [1, 2, 2, 1], pool_type = 'max'):
	"""
	执行 pooling 层的前向运算单元

	Args:
		layer_name: str 类型，表示该 pooling 层的名字
		X: 输入 tensor，维数为 [batch_size, height, width, channels]
		ksize: pooling 层核的大小，在 VGGNet 中使用的是 [1, 2, 2, 1]
		strides: pooling 层的步长，在 VGGNet 中使用的是 [1, 2, 2, 1]
		pool_type: str 类型，表示 pooling 的类型
	
	Returns：
		X: 执行 pooling 运算之后的输出张量
	"""

	if 'max' == pool_type:
		X = tf.nn.max_pool(X, ksize, strides = strides, padding = 'SAME', 
						   name = layer_name + 'max_pool')
	else:
		X = tf.nn.avg_pool(X, ksize, strides = strides, padding = 'SAME', 
						   name = layer_name + 'avg_pool')
	return X

def my_batch_normalization(X, epsilon = 1e-8):
	"""
	执行 batch normalization 过程，加速训练过程。batch Normalization 加在卷积运算之后，
	激活函数之前

	Args:
		X: 输入 tensor，WX + b 的结果

	Returns:
		X: 输出 tensor，执行 BN 之后的结果
	"""

	kernel_num = X.shape[-1]    # 因为 X 是执行卷积运算之后的输出，所以 channels 就是该层卷积核的数目
	shift = tf.Variable(np.zeros([kernel_num]), dtype = tf.float32)
	scale = tf.Variable(np.ones([kernel_num]), dtype = tf.float32)
	batch_mean, batch_var = tf.nn.moments(X, axis = [0, 1, 2])
	X = tf.nn.batch_normalization(X, mean = batch_mean, variance = batch_var, 
								  offset = shift, scale = scale, 
								  variance_epsilon = epsilon)
	return X

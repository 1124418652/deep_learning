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

def my_batch_normalization(X, is_conv = False, epsilon = 1e-8):
	"""
	执行 batch normalization 过程，加速训练过程。batch Normalization 加在卷积运算之后，
	激活函数之前

	Args:
		X: 输入 tensor，WX + b 的结果
		is_conv: boolean 类型的数据，表示当前层是否为卷积层，对于卷积层，需要对 0,1,2 维的数据
				 求均值和方差，对于全连接层，只需要对 0 维的数据求均值和方差
		epsilon: 防止分母为0而加的一个小整数

	Returns:
		X: 输出 tensor，执行 BN 之后的结果
	"""

	kernel_num = X.shape[-1]    # 因为 X 是执行卷积运算之后的输出，所以 channels 就是该层卷积核的数目
	shift = tf.Variable(np.zeros([kernel_num]), dtype = tf.float32)
	scale = tf.Variable(np.ones([kernel_num]), dtype = tf.float32)
	if is_conv:
		batch_mean, batch_var = tf.nn.moments(X, axes = [0, 1, 2])
	else:
		batch_mean, batch_var = tf.nn.moments(X, axes = [0])
	X = tf.nn.batch_normalization(X, mean = batch_mean, variance = batch_var, 
								  offset = shift, scale = scale, 
								  variance_epsilon = epsilon)
	return X

def dense(layer_name, X, num_nodes, batch_norm = True, activate_type = 'relu'):
	"""
	执行全连接层的计算单元

	Args:
		layer_name: str 类型，表示该全连接层的名字
		X: 输入 tensor，维数为 (m, feature_num)
		num_nodes: int 类型，表示该全连接层中节点的数目
		batch_norm: boolean 类型数据，表示是否需要进行 batch normalization
		activate_type: str 类型，表示该层所使用的激活函数的类型，{'relu', 'sigmoid', 'tanh'}
	Returns:
		X: 全连接层的输出 tensor，维数为 (m, num_nodes)
	"""

	assert(2 == len(X.shape))
	num_features = X.shape[-1]
	with tf.variable_scope(layer_name):
		W = tf.get_variable(name='weights', shape=(num_features, num_nodes),
							dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable(name='biases', shape=(num_nodes), dtype=tf.float32,
							initializer=tf.constant_initializer(0.0))
		X = tf.matmul(X, W)
		X = tf.nn.bias_add(X, b, name = 'Z')

		# 按要求进行batch normalization
		if batch_norm:
			X = my_batch_normalization(X, False)

		if 'relu' == activate_type:
			X = tf.nn.relu(X, name = 'relu_activate')
		elif 'sigmoid' == activate_type:
			X = tf.nn.sigmoid(X, name = 'sigmoid_activate')
		elif 'tanh' == activate_type:
			X = tf.nn.tanh(X, name = 'tanh_activate')
		return X

def loss_calculate(logits, labels, loss_type = 'cross_entropy'):
	"""
	计算前向传播中损失函数的结果
	
	Args:
		logits: 最后一层全连接层的输出
		labels: 每个样本对应的标签
		loss_type: 损失函数的类型，{'cross_entropy', 'L2'}
	"""

	with tf.name_scope('loss') as scope:
		if 'cross_entropy' == loss_type:
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
			loss = tf.reduce_mean(cross_entropy, name='loss')
			tf.summary.scalar(scope + '/loss' ,loss)
		elif 'L2' == loss_type:
			loss = tf.reduce_mean(tf.square(logits - labels), keep_dims=False)
			tf.summary.scalar(scope+'/loss', loss)
		return loss

def accuracy_calculate(logits, labels):
	"""
	计算验证集的准确性
	"""

	with tf.name_scope('accuracy') as scope:
		correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
		correct = tf.cast(correct, dtype = tf.float32)
		accuracy = tf.reduce_mean(correct) * 100.0
		tf.summary.scalar(scope+'accuracy', accuracy)
		return accuracy

def optimizer(loss, global_step, optimizer_type = 'Adam', lr = 1e-4):
	"""
	设置网络的优化器

	Args:
		loss: 计算得到的损失函数
		global_step: tf.Variable 类型的对象，用于记录迭代优化的次数，主要用于参数的输出和保存
		optimizer_type: str类型，设置优化器的类型，{'Adam', 'SGD'}
		lr: 设置网络的学习率
	"""

	if optimizer_type == 'Adam':
		train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
	elif optimizer_type == 'SGD':
		train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step)
	return train_step

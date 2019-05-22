#-*- coding: utf-8 -*-
"""
Created on 2019/5/13

@Author: xhj
"""

import utils
import numpy as np
import tensorflow as tf


def VGG16(X, n_classes = 10, is_pretrain = True):
	"""
	实现 VGG16 的网络模型

	Args:
		X: 输入 tensor，维数为 (batch_num, height, width, channels)
		n_classes: int 类型，模型最后一层 softmax 层的输出节点数，即分类的类别数
		is_pretrain: boolean 类型，表示参数是否需要训练
	"""

	with tf.name_scope('VGG16') as scope:
		
		# 第一组卷积
		X = utils.conv('Conv1_1', X, 64, is_pretrain = is_pretrain)
		X = utils.conv('Conv1_2', X, 64, is_pretrain = is_pretrain)
		with tf.name_scope('pool1'):
			X = utils.pool('pool1', X)

		# 第二组卷积
		X = utils.conv('Conv2_1', X, 128, is_pretrain = is_pretrain)
		X = utils.conv('Conv2_2', X, 128, is_pretrain = is_pretrain)
		with tf.name_scope('pool2'):
			X = utils.pool('pool2', X)

		# 第三组卷积
		X = utils.conv('Conv3_1', X, 256, is_pretrain = is_pretrain)
		X = utils.conv('Conv3_2', X, 256, is_pretrain = is_pretrain)
		X = utils.conv('Conv3_3', X, 256, is_pretrain = is_pretrain)
		with tf.name_scope('pool3'):
			X = utils.pool('pool3', X)

		# 第四组卷积
		X = utils.conv('Conv4_1', X, 512, is_pretrain = is_pretrain)
		X = utils.conv('Conv4_2', X, 512, is_pretrain = is_pretrain)
		X = utils.conv('Conv4_3', X, 512, is_pretrain = is_pretrain)
		with tf.name_scope('pool4'):
			X = utils.pool('pool4', X)

		# 第五组卷积
		X = utils.conv('Conv5_1', X, 512, is_pretrain = is_pretrain)
		X = utils.conv('Conv5_2', X, 512, is_pretrain = is_pretrain)
		X = utils.conv('Conv5_3', X, 512, is_pretrain = is_pretrain)
		with tf.name_scope('pool5'):
			X = utils.pool('pool5', X)

		feature_num = np.prod(X.shape[1:])
		X = tf.reshape(X, shape = (-1, feature_num))     # 将最后一层的feature map 拉平
		
		# 第一层全连接层
		X = utils.dense('fc6', X, num_nodes = 4096)
		# batch normalization 应该放在 WX+b 之后进行
		# with tf.name_scope('batch_normal1'):
		# 	X = utils.my_batch_normalization(X)

		# 第二层全连接层
		X = utils.dense('fc7', X, num_nodes = 4096)
		# with tf.name_scope('batch_normal2'):
		# 	X = utils.my_batch_normalization(X)

		# 第三层全连接层
		X = utils.dense('fc8', X, num_nodes = n_classes)

		return X
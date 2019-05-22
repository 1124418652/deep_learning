#-*- coding: utf-8 -*-
"""
Created on 2019/5/13

@Author: xhj
"""

import os
import numpy as np
import tensorflow as tf 


IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNRL = 3
LABEL_BYTES = 1
IMG_BYTES = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNRL


def read_cifar10(data_dir, is_train, batch_size, shuffle = True):
	"""
	通过此函数读取 CIFAR 数据集的训练集或测试集

	Args:
		data_dir: 数据集所在的文件夹路径
		is_train: boolean 类型，表示是否为训练集
		batch_size: int 类型，表示 batch的大小
		shuffle: boolean 类型，表示在构建 mini batch 时是否需要打乱数据的顺序
	Returns:
		images: 包含一个 mini batch 的图片 tensor，维数为 (batch_num, height, width, channels)
		labels: mini batch 中图片所对应的标签
	"""

	num_classes = 10

	with tf.name_scope('input') as scope:
		if not os.path.split(data_dir)[-1] == 'cifar-10-batches-bin':
			data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
		if is_train:
			filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in np.arange(1, 6)]
		else:
			filenames = [os.path.join(data_dir, 'test_batch.bin')]

		try:
			filename_queue = tf.train.input_producer(filenames)    # 构建文件名队列
			# 从文件中读取指定长度的位，即一个样本，标签占第一个byte，对应的图片占之后的 3072 bytes
			reader = tf.FixedLengthRecordReader(LABEL_BYTES + IMG_BYTES)
			key, value = reader.read(filename_queue)
			record_bytes = tf.decode_raw(value, tf.uint8)

			label = tf.slice(record_bytes, [0], [LABEL_BYTES])     # 从数据的第0个元素开始获取标签的切片
			label = tf.cast(label, tf.int32)

			image_raw = tf.slice(record_bytes, [LABEL_BYTES], [IMG_BYTES])   # 从数据中获取图片数据的切片
			image_raw = tf.reshape(image_raw, [IMG_CHANNRL, IMG_HEIGHT, IMG_WIDTH])
			image = tf.transpose(image_raw, (1, 2, 0))             # 交换数据的维度，在原始数据中channel通道位于前面
			image = tf.cast(image, tf.float32)

			# 对图片数据进行标准化处理，(x-mean) / var
			image = tf.image.per_image_standardization(image)      # vgg 唯一的图片增强处理就是图片的标准化

			# images 表示的是每个batch的图片数据，batch_label表示的是每个batch的标签数据
			if shuffle:
				images, label_batch = tf.train.shuffle_batch([image, label],
															  batch_size = batch_size,
															  capacity = 20000,
															  min_after_dequeue = 300,
															  num_threads = 64)
			else:
				images, label_batch = tf.train.batch([image, label],
													  batch_size = batch_size,
													  capacity = 20000,
													  num_threads = 64)
			label_batch = tf.one_hot(label_batch, depth = num_classes)         # 将标签转换成 one-hot 形式
			label_batch = tf.cast(label_batch, dtype = tf.int32)
			label_batch = tf.reshape(label_batch, [batch_size, num_classes])

		except:
			raise ValueError("Can't load dataset")

		return images, label_batch

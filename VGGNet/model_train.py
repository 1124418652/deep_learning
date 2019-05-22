#-*- coding: utf-8 -*-
"""
Create on 2019/5/13

@Author: xhj
"""

import os
import math
import utils
import input_data
import numpy as np 
import tensorflow as tf
from vgg16 import VGG16


IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNRL = 3
NUM_CLASSES = 10

def train(batch_size, epochs, shuffle):
	"""
	训练模型的参数

	Args:
		batch_size: 训练数据集的批次大小
		epochs: 训练所需要的迭代次数
		shuffle: 表示在构建 mini batch 时是否需要进行顺序打乱
	"""

	data_dir = '../../datasets/CIFAR10/'
	log_dir = 'logs2'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	train_log_dir = 'logs2/train/'
	val_log_dir = 'logs2/validation/'
	if not os.path.exists(train_log_dir):
		os.makedirs(train_log_dir)
	if not os.path.exists(val_log_dir):
		os.makedirs(val_log_dir)

	# 获取训练数据集合测试数据集
	train_image_batch, train_label_batch = input_data.read_cifar10(data_dir, 
		True, batch_size, shuffle)
	val_image_batch, val_label_batch = input_data.read_cifar10(data_dir, 
		False, batch_size, shuffle)

	X = tf.placeholder(dtype = tf.float32, shape = [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNRL])
	_y = tf.placeholder(dtype = tf.int32, shape = [None, NUM_CLASSES])
	
	# 获取模型前向传播至softmax层的输出
	logits = VGG16(X, n_classes = 10, is_pretrain = True)
	loss = utils.loss_calculate(logits, _y)
	accuracy = utils.accuracy_calculate(logits, _y)

	# 记录网络迭代优化的次数
	my_global_step = tf.Variable(0, trainable = False, name = 'global_step')
	train_op = utils.optimizer(loss, my_global_step)

	saver = tf.train.Saver(tf.global_variables())
	summary_op = tf.summary.merge_all()

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		coord = tf.train.Coordinator()

		# 启动入队线程，函数返回线程ID的列表
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

		try:
			for step in np.arange(epochs):
				# 检查是否应该终止所有线程，当文件名队列中的所有文件都已经出队，会抛出一个
				# OutOfRangeError的异常，跳转到 except 和 finally 的程序块执行，在下一个循环中
				# 就会进入该 if 语句块中
				if coord.should_stop():
					break

				train_images, train_labels = sess.run([train_image_batch, train_label_batch])
				_, train_loss, train_accuracy = sess.run([train_op, loss, accuracy],
					feed_dict = {X: train_images, _y: train_labels})

				if step % 50 == 0 or (step + 1) == epochs:
					print("Step: %d, loss: %.4f, accuracy: %.4f%%" % (step, train_loss, train_accuracy))
					summary_str = sess.run(summary_op, feed_dict = {X: train_images, _y: train_labels})
					train_summary_writer.add_summary(summary_str, step)

		except tf.errors.OutOfRangeError:
			print('Finish training!')
		finally:
			coord.request_stop()

		# 把通过 tf.train.start_queue_runners() 开启的线程加入主线程，防止主线程运行结束后
		# 直接退出
		coord.join(threads)


if __name__ == '__main__':
	train(64, 1000, True)
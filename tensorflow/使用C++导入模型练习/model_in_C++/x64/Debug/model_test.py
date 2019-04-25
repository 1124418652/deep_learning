# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf 

def recognize(imgname):
	
	if not os.path.exists(imgname):
		raise ValueError("Don't have this file.")

	img = cv2.imread(imgname)
	if not isinstance(img, np.ndarray):
		raise ValueError("Can't open this image.")

	img = cv2.resize(img, (64, 64)) / 255

	with tf.Graph().as_default() as g:
		print(1)
		output_graph_def = tf.GraphDef()
		pb_file_path = 'model_trained.pb'

		print(2)
		with open(pb_file_path, 'rb') as f:
			print(2)
			output_graph_def.ParseFromString(f.read())
			print(3)
			# 将计算图从 output_graph_def 导入到当前的默认图中
			tf.import_graph_def(output_graph_def, name = '')      

		print(2)
		with tf.Session() as sess:
			tf.global_variables_initializer().run()
			input_x = sess.graph.get_tensor_by_name('input:0')         # 获取张量
			prediction = sess.graph.get_tensor_by_name('output:0') 
			
			pre = sess.run(prediction, feed_dict = {input_x: [img]})           # 将图片喂入网络进行测试

			print(pre)
			

#recognize('../../model_in_C++/test.jpg')


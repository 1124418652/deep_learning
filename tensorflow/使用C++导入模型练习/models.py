#-*- coding: utf-8 -*-

import os
import h5py
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.framework import graph_util


def read_and_decode_tfrecord_files(filename, batch_size):
    """
    从 tfrecord 格式的文件中读取出数据，并将其转换成正常的图片和标签
    
    Args:
        filename: tfrecords 文件的文件名（完整的路径）
        batch_size: 批大小
    Returns:
        image_batch: 图片数据 batch
        label_batch: label 数据 batch
    """
    
    filename_queue = tf.train.string_input_producer([filename])   # 创建一个文件名队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized_example, 
                        features = {'label': tf.FixedLenFeature([], tf.int64),
                                    'image_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.decode_raw(img_features['image_raw'], tf.uint8)    # 从二进制解码到 uint8
    image = tf.reshape(image, [64, 64, 3])            # 调整维度
    image = tf.cast(image, tf.float32) / 255.0          # 数据格式转换
    label = tf.cast(img_features['label'], tf.int32)
#     label = tf.reshape(label, [-1, 1])
    
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size = batch_size,    # batch的大小
                                              num_threads = 64,    # 线程数
                                              capacity = 2000)     # 队列中最多能有多少数据
    return image_batch, label_batch

def my_batch_norm(inputs):
    """
    对输入的数据进行batch normalization
    
    Args：
        inputs：inputs不是上一层的输出，而是 Wx+b，其中 x 才是上一层的输出，
                这就解释了为什么 BN 在求均值和方差时是对 [0,1,2] 维的数据进
                行求解。
    Returns:
        inputs: 输入的数据
        batch_mean: batch 内的均值
        batch_var: batch 内的方差
        beta, scale: 需要训练的权重值和偏差值
    """
    
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]), dtype = tf.float32)
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), dtype = tf.float32)
    batch_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable = False)
    batch_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable = False)
    
    batch_mean, batch_var = tf.nn.moments(inputs, axes = [0, 1, 2])
    
    return inputs, batch_mean, batch_var, beta, scale


def build_network():
    """
    搭建完整的网络结构
    """
    
    # 网络的输入
    x = tf.placeholder(tf.float32, shape = [None, 64, 64, 3], name = 'input')
    y = tf.placeholder(tf.int32, shape = [None, 1], name = 'input_label')
    lr = tf.placeholder(tf.float32)     # 网络反向传播的学习率，在迭代过程中需要动态改变
    
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial, name = name)
    
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial, name = name)
    
    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding = 'SAME')
    
    def pool(x, pool_type = 'max'):
        if 'max' == pool_type:
            return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], 
                                  strides=[1, 2, 2, 1],
                                  padding = 'SAME')
        else:
            return tf.nn.avg_pool(x, ksize = [1, 2, 2, 1],
                                  strides = [1, 2, 2, 1],
                                  padding = 'SAME')
    
    # 第一组卷积，包含两个卷积层
    with tf.name_scope('conv1_1') as scope:
        # 该层没有池化层
        W_conv1 = weight_variable([3, 3, 3, 64], 'W_conv1')
        b_conv1 = bias_variable([64], 'b_conv1')
        Z_conv1 = tf.nn.bias_add(conv2d(x, W_conv1), b_conv1, name = 'Z_conv1')
        
        # 获取均值和方差
        inputs, batch_mean, batch_var, beta, scale = my_batch_norm(Z_conv1)
        
        """
        batch normalization 公式：
        x = (x - batch_mean) / (sqrt(batch_var) + 0.001)
        x_out = x * scale + beta
        """
        conv_batch_norm = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
        A_conv1 = tf.nn.relu(conv_batch_norm, name = 'A_conv1')
        
    with tf.name_scope('conv1_2') as scope:
        W_conv2 = weight_variable([3, 3, 64, 64], 'W_conv2')
        b_conv2 = bias_variable([64], 'b_conv2')
        Z_conv2 = tf.nn.bias_add(conv2d(A_conv1, W_conv2), b_conv2, name = 'Z_conv2')
        
        # batch normalization
        inputs, batch_mean, batch_var, beta, scale = my_batch_norm(Z_conv2)
        conv_batch_norm = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
        A_conv2 = tf.nn.relu(conv_batch_norm, name = 'A_conv2')
        pool1 = pool(A_conv2)   # 结束完一组卷积之后进行池化
        
    # 第二组卷积，包含两个卷积层
    with tf.name_scope('conv2_1') as scope:
        W_conv3 = weight_variable([3, 3, 64, 128], 'W_conv3')
        b_conv3 = bias_variable([128], 'b_conv3')
        Z_conv3 = tf.nn.bias_add(conv2d(pool1, W_conv3), b_conv3, name = 'Z_conv3')
        inputs, batch_mean, batch_var, beta, scale = my_batch_norm(Z_conv3)
        conv_batch_norm = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
        A_conv3 = tf.nn.relu(conv_batch_norm, name = 'A_conv3')
        
    with tf.name_scope('conv2_2') as scope:
        W_conv4 = weight_variable([3, 3, 128, 128], 'W_conv4')
        b_conv4 = bias_variable([128], 'b_conv4')
        Z_conv4 = tf.nn.bias_add(conv2d(A_conv3, W_conv4), b_conv4, name = 'Z_conv4')
        inputs, batch_mean, batch_var, beta, scale = my_batch_norm(Z_conv4)
        conv_batch_norm = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
        A_conv4 = tf.nn.relu(conv_batch_norm, name = 'A_conv4')
        pool2 = pool(A_conv4)
        
    # 第三组卷积，包含两个卷积层
    with tf.name_scope('conv3_1') as scope:
        W_conv5 = weight_variable([3, 3, 128, 256], 'W_conv5')
        b_conv5 = bias_variable([256], 'b_conv5')
        Z_conv5 = tf.nn.bias_add(conv2d(pool2, W_conv5), b_conv5, name = 'Z_conv5')
        inputs, batch_mean, batch_var, beta, scale = my_batch_norm(Z_conv5)
        conv_batch_norm = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
        A_conv5 = tf.nn.relu(conv_batch_norm, name = 'A_conv5')
        
    with tf.name_scope('conv3_2') as scope:
        W_conv6 = weight_variable([3, 3, 256, 256], 'W_conv6')
        b_conv6 = bias_variable([256], 'b_conv6')
        Z_conv6 = tf.nn.bias_add(conv2d(A_conv5, W_conv6), b_conv6, name = 'Z_conv6')
        inputs, batch_mean, batch_var, beta, scale = my_batch_norm(Z_conv6)
        conv_batch_norm = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, 0.001)
        A_conv6 = tf.nn.relu(conv_batch_norm, name = 'A_conv6')
        pool3 = pool(A_conv6)
        
    # 第四组卷积
    # 第五组卷积
    
    # fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool3.get_shape()[1:]))
        W_fc6 = weight_variable([shape, 128], 'W_fc6')
        b_fc6 = bias_variable([128], 'b_fc6')
        pool_flat = tf.reshape(pool3, [-1, shape])
        A_fc6 = tf.nn.relu(tf.matmul(pool_flat, W_fc6) + b_fc6, name = 'A_fc6')
        
    # fc7
    with tf.name_scope('fc7') as scope:
        W_fc7 = weight_variable([128, 64], 'W_fc7')
        b_fc7 = bias_variable([64], 'b_fc7')
        A_fc7 = tf.nn.relu(tf.matmul(A_fc6, W_fc7) + b_fc7, name = 'A_fc7')
        
    # fc8
    with tf.name_scope('fc8') as scope:
        W_fc8 = weight_variable([64, 1], 'W_fc8')
        b_fc8 = bias_variable([1], 'b_fc8')
        A_fc8 = tf.nn.sigmoid(tf.matmul(A_fc7, W_fc8) + b_fc8, name = 'A_fc8')
        
    # calculate cost and optimizer the model
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y,tf.float32), 
                                                                  logits=A_fc8))
    predict = tf.cast(A_fc8 > 0.5, tf.int32, name = 'output')      # 给出预测值
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), tf.float32))            # 计算预测的准确率   
    train_step = tf.train.AdamOptimizer(lr).minimize(cost)     # 更新模型参数
    
    return dict(x = x, 
                y = y,
                lr = lr,
                cost = cost, 
                predict = predict,
                accuracy = accuracy,
                train_step = train_step)


def train_network(graph, batch_size, num_epoches, pd_file_path):
    """
    对传入该函数的网络进行训练
    
    Args:
        graph: 通过 build_network 函数构建的计算图
        batch_size: 批大小
        num_epoches: 网络迭代的次数
        pd_file_path: pd 文件保存的路径
    """
    
    image_batch, label_batch = read_and_decode_tfrecord_files(filename = 'train.tfrecords',
                                                              batch_size = batch_size)
    val_image_batch, val_label_batch = read_and_decode_tfrecord_files(filename = 'val.tfrecords',
                                                                      batch_size = batch_size)
    init = tf.global_variables_initializer()     # 变量初始化
    
    with tf.Session() as sess:                   # 创建会话
        sess.run(init)
        
        """
        在 tensorflow 中，当文件名队列中的元素取完之后，会抛出一个 OutofRangeError 异常
        """
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)   # 启动内存队列
        try:
            for epoch in range(num_epoches):     # 执行模型的迭代更新，并检查是否出现异常
                train_data, train_label = sess.run([image_batch, label_batch])
                val_data, val_label = sess.run([val_image_batch, val_label_batch])
                train_label = train_label.reshape([-1, 1])
                val_label = val_label.reshape([-1, 1])
                cost_val, accuracy_train, _ , prediction= sess.run([graph['cost'],
                                                        graph['accuracy'],
                                                        graph['train_step'],
                                                        graph['predict']],
                                                       feed_dict = {graph['x']: train_data,
                                                                    graph['y']: train_label,
                                                                    graph['lr']: 1e-5})
                accuracy_val = sess.run(graph['accuracy'], feed_dict = {
                    graph['x']: val_data,
                    graph['y']: val_label
                    })
                print('Cost of Iter',epoch,': ',cost_val)
                print('Train accuracy: ',accuracy_train)
                print('Validation accuracy: ', accuracy_val)
                print(prediction.flatten())
                print(train_label.flatten())

                max_val_accuracy = 0.8
                max_train_accuracy = 0.9
                if accuracy_val > max_val_accuracy and accuracy_train > max_train_accuracy:
                    max_val_accuracy = accuracy_val
                    max_train_accuracy = accuracy_train
                    constant_graph = graph_util.convert_variables_to_constants(sess, 
                        sess.graph_def, ['output'])
                    with tf.gfile.FastGFile(pd_file_path, mode = 'wb') as f:
                        f.write(constant_graph.SerializeToString())
                
        except tf.errors.OutOfRangeError:
            print('Training has finished')
        finally:
            coord.request_stop()
        
        coord.join(threads)         # 把开启的线程加入主线程，防止主线程运行结束之后直接退出
        print('All threads are stopped!')
        

graph = build_network()
train_network(graph, 50, 150, 'model_trained.pb')
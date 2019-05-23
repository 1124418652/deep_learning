#-*- coding: utf-8 -*-
"""
Create on 2019/5/23

@Author: xhj
"""
import cv2
import numpy as np 
import tensorflow as tf


class Yolo(object):
    """
    Yolo架构分为两步份，卷积部分负责图片特征的提取，全连接部分负责类别判断以及
    目标框的回归，每张图片经过网络后的输出是7x7的网格，在论文中每个网格是一个
    30维的向量：
    [confidence, x_center, y_center, w, h] x 2 + [conditional_probability] x 20
    """
    
    def __init__(self):

        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]
        self.num_classes = len(self.classes)
      
        """
         目标中心坐标的预测值(x,y)是偏移量坐标，即相对于所在grid的偏移值，
         需要加上网格的抵消量，如第2行第3列的grid的一个bbox坐标预测是
         (0.2, 0.5)，需要通过坐标变换(1+0.2, 2+0.5)/7转换成原图中的归一化
         坐标。
        """
        self.y_offset, self.x_offset = np.mgrid[0:7, 0:7]     # 计算x和y坐标的抵消量
        self.confidence_thresh = 0.2
        self.iou_thresh = 0.5
        self.max_output_size = 10
        self.img_shape = (448, 448)
        self.batch_size = 56
        self.coord_scale = 5.
        self.noobject_scale = 0.5
        self.object_scale = 1.
        self.class_scale = 2.

    def build_network(self, faster_Yolo = False):
        """
        使用googLeNet作为基础网络构建网络架构
        """

        net_input = tf.placeholder(tf.float32, [None, 448, 448, 3])
        with tf.variable_scope('yolo'):
            with tf.name_scope('group1'):
                X = self._conv('Conv1', X, 64, 7, 2)
                X = self._pool('pool1', X, 'max', 2, 2)
            with tf.name_scope('group2'):
                X = self._conv('Conv2', X, 192, 3, 1)
                X = self._pool('pool2', X, 'max', 2, 2)
            with tf.name_scope('group3'):
                X = self._conv('Conv3', X, 128, 1, 1)
                X = self._conv('Conv4', X, 256, 3, 1)
                X = self._conv('Conv5', X, 256, 1, 1)
                X = self._conv('Conv6', X, 512, 3, 1)
                X = self._pool('pool3', X, 'max', 2, 2)
            with tf.name_scope('group4'):
                X = self._conv('Conv7', X, 256, 1, 1)
                X = self._conv('Conv8', X, 512, 3, 1)
                X = self._conv('Conv9', X, 256, 1, 1)
                X = self._conv('Conv10', X, 512, 3, 1)
                X = self._conv('Conv11', X, 256, 1, 1)
                X = self._conv('Conv12', X, 512, 3, 1)
                X = self._conv('Conv13', X, 256, 1, 1)
                X = self._conv('Conv14', X, 512, 3, 1)
                X = self._conv('Conv15', X, 512, 1, 1)
                X = self._conv('Conv16', X, 1024, 3, 1)
                X = self._pool('pool4', X, 'max', 2, 2)
            with tf.name_scope('group5'):
                X = self._conv('Conv17', X, 512, 1, 1)
                X = self._conv('Conv18', X, 1024, 3, 1)
                X = self._conv('Conv19', X, 512, 1, 1)
                X = self._conv('Conv20', X, 1024, 3, 1)
                X = self._conv('Conv21', X, 1024, 3, 1)
                X = self._conv('Conv22', X, 1024, 3, 2)
            with tf.name_scope('group6'):
                X = self._conv('Conv23', X, 1024, 3, 1)
                X = self._conv('Conv24', X, 1024, 3, 1)
            X = self._flatten(X)
            with tf.name_scope('Fully_connect'):
                X = self._dense('fc1', X, 4096, tf.nn.leaky_relu) 
                X = self._dense('fc2', X, 1470, tf.nn.leaky_relu)
            
            return net_input, X

    def _conv(self, layer_name, X, kernel_num, ksize, stride):
        """
        执行前向卷积运算
        """

        in_channel = X.get_shape().as_list()[-1]
        with tf.variable_scope(layer_name):
            W = tf.Variable(tf.truncated_normal([ksize, ksize, in_channel, kernel_num], stddev=0.1),
                            name='weights')
            b = tf.Variable(tf.zeros([kernel_num]), name="biases")
            pad_size = ksize // 2 
            pad_mat = np.array([[0, 0], [pad_size, pad_size], 
                                [pad_size, pad_size], [0, 0]])
            X = tf.pad(X, pad_mat)
            X = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='VALID',
                             name='Z')
            X = tf.nn.leaky_relu(tf.nn.bias_add(X, b), alpha=0.01, name='A')
            return X

    def _dense(self, layer_name, X, num_nodes, activation=None):
        """
        执行前向全连接运算
        """

        assert(2 == len(X.shape))
        num_features = X.shape.as_list()[-1]

        with tf.variable_scope(layer_name):
            W = tf.Variable(tf.truncated_normal([num_features, num_nodes],
                stddev=0.1), name='weights')
            b = tf.Variable(tf.zeros([num_nodes]), name='biases')
            X = tf.nn.xw_plus_b(X, W, b, name='Z')
            if activation:
                X = activation(X, name='A')
            return X

    def _pool(self, layer_name, X, type='max', psize=2, stride=2):
        """
        执行前向的池化运算
        """

        with tf.variable_scope(layer_name):
            if 'max' == type:
                X = tf.nn.max_pool(X, [1, psize, psize, 1], 
                        [1, stride, stride, 1], padding='SAME', name='max_pool')
            else:
                X = tf.nn.avg_pool(X, [1, psize, psize, 1], 
                        [1, stride, stride, 1], padding='SAME', name='avg_pool')
            return X

    def _flatten(self, X):
        """
        在卷积层转池化层之前将张量进行拉伸
        """

        X = tf.transpose(X, [0, 3, 1, 2])    # channel first mode
        num_features = np.prod(X.shape[1:])
        return tf.reshape(X, [-1, num_features])


if __name__ == '__main__':
    yolo = Yolo()
    yolo.build_network()
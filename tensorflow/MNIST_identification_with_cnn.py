import os
import re
import struct
import tensorflow as tf
import numpy as np

# 获取数据
dataset_dir = 'datasets/mnist_database/'
train_set_path = os.path.join(dataset_dir, 'train-images.idx3-ubyte')
train_label_path = os.path.join(dataset_dir, 'train-labels.idx1-ubyte')
test_set_path = os.path.join(dataset_dir, 't10k-images.idx3-ubyte')
test_label_path = os.path.join(dataset_dir, 't10k-labels.idx1-ubyte')

def load_data(file_path):
    
    if not os.path.exists(file_path):
        print("The file is not exist!")
        return
    
    fr_binary = open(file_path, 'rb')
    buffer = fr_binary.read()

    if re.search('\w+-(images)\.', os.path.split(file_path)[-1]) is not None:
        """
        提取数据文件
        """
        head = struct.unpack_from('>IIII', buffer, 0)
        offset = struct.calcsize('>IIII')       # 定位到字节流中 data 开始的位置
        img_num, width, height = head[1:]   
        format_str = '>{0}B'.format(img_num * width * height)
        data = struct.unpack_from(format_str, buffer, offset)
        fr_binary.close()
        data = np.reshape(data, [img_num, width, height])
        return data
    
    elif re.search('\w+-(labels)\.', file_path) is not None:
        """
        提取标签文件
        """
        head = struct.unpack_from('>II', buffer, 0)
        label_num = head[1]
        offset = struct.calcsize('>II')
        format_str = '>{0}B'.format(label_num)
        labels = struct.unpack_from(format_str, buffer, offset)
        fr_binary.close()
        labels = np.reshape(labels, [label_num])
        return labels
        
train_img = load_data(train_set_path)
train_label = load_data(train_label_path)
test_img = load_data(test_set_path)
test_label = load_data(test_label_path)

# 调整数据结构
def convert_one_hot(y, num_classes):
    
    y_one_hot = np.zeros((y.shape[-1], num_classes))
    for i in range(y.shape[-1]):
        y_one_hot[i][y[i]] = 1
    return y_one_hot
    
train_label_one_hot = convert_one_hot(train_label, 10)
test_label_one_hot = convert_one_hot(test_label, 10)
train_data = (np.reshape(train_img, (60000, 784)) / 255).astype(np.float32)
test_data = (np.reshape(test_img, (10000, 784)) / 255).astype(np.float32)

# 生成batches
def get_batches(X, y, batch_size, axis = 0, seed = 0):
    
    assert(X.shape[axis] == y.shape[axis])
    np.random.seed(seed)
    m = X.shape[axis]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    num_complete_minibatches = m // batch_size
    
    if 0 == axis:
        shuffled_X = X[permutation, :]
        shuffled_y = y[permutation, :]
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[k * batch_size: (k + 1) * batch_size, :]
            mini_batch_y = shuffled_y[k * batch_size: (k + 1) * batch_size, :]
            mini_batches.append((mini_batch_X, mini_batch_y))
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * batch_size, :]
            mini_batch_y = shuffled_y[num_complete_minibatches * batch_size, :]
            mini_batches.append((mini_batch_X, mini_batch_y))
        return mini_batches
        
    elif 1 == axis:
        shuffled_X = X[:, permutation]
        shuffled_y = y[:, permutation]
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * batch_size: (k + 1) * batch_size]
            mini_batch_y = shuffled_y[:, k * batch_size: (k + 1) * batch_size]
            mini_batches.append((mini_batch_X, mini_batch_y))
        if m % batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * batch_size]
            mini_batch_y = shuffled_y[:, num_complete_minibatches * batch_size]
            mini_batches.append((mini_batch_X, mini_batch_y))
        return mini_batches

# 构建卷积网络

def weight_variable(shape):
    # 初始化权重矩阵
    initial = tf.truncated_normal(shape, stddev = 0.1)     # 生成一个截断的正态分布
    return tf.Variable(initial)

def bias_variable(shape):
    # 初始化偏置矩阵
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME'):
    """
    Calculate the convolve of x and W
    
    Arguments:
    x -- input tensor of shape (batch_num, height, width, channels)
    W -- tensorflow Variable, with the shape of 
         (filter_height, filter_width, in_channels, out_channels)
    stride -- python list, respect to the stride of every dimensions
    padding -- the type of padding, {'SAME', 'VALID'}
         
    Returns:
    tensor after convolve
    """
    
    return tf.nn.conv2d(x, W, strides = strides, padding = padding)

def pool(x, pool_type = 'max', ksize = [1, 2, 2, 1], 
         strides = [1, 2, 2, 1], padding = 'SAME'):
    """
    the pool layer of convolution network
    
    Arguments:
    x -- input tensor of shape (batch_num, height, width, channels)
    mode -- the type of pool layer {'max', 'average'}
    ksize -- A list or tuple of 4 ints. The size of the window for each dimension
             of the input tensor.
    strides -- A list or tuple of 4 ints. The strides of the sliding window for
               each dimension of the input tensor.
    padding -- the type of padding, {'SAME', 'VALID'}
    """
    
    if 'max' == pool_type:
        return tf.nn.max_pool(x, ksize = ksize, strides = strides, padding = padding)
    elif 'average' == pool_type:
        return tf.nn.avg_pool(x, ksize = ksize, strides = strides, padding = padding)
    
# 定义网络的输入
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])    # reshape the input x


"""
定义网络的结构: 
conv + relu + max_pool + conv + relu + max_pool + full_connection(relu) 
+ softmax
"""
# layer1,input dimension (-1, 28, 28, 1), output dimension (-1, 14, 14, 32)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
a_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
a_conv1_pool = pool(a_conv1)

# layer2, input dimension (-1, 14, 14, 32), output dimension (-1, 7, 7, 64)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
a_conv2 = tf.nn.relu(conv2d(a_conv1_pool, W_conv2) + b_conv2)
a_conv2_pool = pool(a_conv2)

# full connection layer1, reshape the dimension of input first
keep_prob = tf.placeholder(tf.float32)
A_input = tf.reshape(a_conv2_pool, [-1, 7 * 7 * 64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])     # 第一层全连接层有1024个神经元
b_fc1 = bias_variable([1024])
Z_fc1 = tf.matmul(A_input, W_fc1) + b_fc1
A_fc1 = tf.nn.relu(Z_fc1)
A_fc1_prob = tf.nn.dropout(A_fc1, keep_prob)

# full connection layer2, softmax layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
Z_fc2 = tf.matmul(A_fc1_prob, W_fc2) + b_fc2
prediction = tf.nn.softmax(Z_fc2)

# calculate cross entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction))
# 使用 Adam 进行网络参数的优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 计算准确率
correct_prediction = tf.equal(tf.argmax(prediction, axis = 1), tf.argmax(y, axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())       # 初始化网络中的所有变量
    mini_batches = get_batches(train_data, train_label_one_hot, batch_size=100)
    for epoch in range(21):
        for X_data, y_data in mini_batches:
            sess.run(train_step, feed_dict = {x: X_data, y: y_data, keep_prob: 0.7})
        acc = sess.run(accuracy, feed_dict = {x: test_data, y: test_label_one_hot, keep_prob: 1})
        print('Iter ' + str(epoch) + ' , Testing Accuracy = ' + str(acc))
        
#-*- coding: utf-8 -*-
"""
Create on 2019/5/22

@Author: xhj
"""

import cv2
import h5py
import time
import data_prepare
import numpy as np 
import tensorflow as tf
from PIL import Image


def data2tfrecord(X, y, filename):
	"""
	将内存中的数据制作为tfrecord格式的数据，tfrecord是谷歌推荐的一种
	二进制文件格式，理论上可以保存任何格式的信息

	Args:
		X: 输入的样本数据
		y: 输入样本对应的标签
		filename: tfrecord格式的文件名
	"""

	writer = tf.python_io.TFRecordWriter(filename)
	for index, image in enumerate(X):
		image = Image.fromarray(image)
		image = image.tobytes()      # 将图片转成二进制格式
		feature = {}

		feature['image_raw'] = tf.train.Feature(bytes_list = tf.train.BytesList(value = [image]))
		feature['label'] = tf.train.Feature(int64_list = tf.train.Int64List(value = [int(y[index])]))
		tf_features = tf.train.Features(feature = feature)       # 将每次迭代的feature合并成features
		tf_example = tf.train.Example(features = tf_features)    # 生成example模块
		tf_serialized = tf_example.SerializeToString()     # 将example编码成二进制
		writer.write(tf_serialized)      # 写入文件

	writer.close()
	print("fininshing generating tfrecord file: ", filename)


def read_and_decode_tfrecord_files(filename, batch_size, shuffle = True, 
								   one_hot = True, classes = 28):
	"""
	从tfrecord格式的文件中提取出数据，并将其从二进制解码，按batch_size制作成数据集

	Args:
		filename: tfrecord 文件的文件名（完整的路径）
		batch_size: 批大小
		ont_hot: boolean类型，表示是否需要使用one-hot格式的标签
		classes: int类型，表示类别数
	Returns:
		image_batch: 图片数据batch
		label_batch: 标签数据batch
	"""

	filename_deque = tf.train.string_input_producer([filename])
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_deque)
	img_features = tf.parse_single_example(serialized_example,
		features = {'label': tf.FixedLenFeature([], tf.int64),
					'image_raw': tf.FixedLenFeature([], tf.string)})

	image = tf.decode_raw(img_features['image_raw'], tf.uint8)     # 从二进制解码到uint8
	image = tf.reshape(image, [28, 28, 1])
	image = tf.cast(image, tf.float32)
	image = tf.image.per_image_standardization(image)    # 将图片做标准化处理

	label = tf.cast(img_features['label'], tf.int32)

	if shuffle:
		image_batch, label_batch = tf.train.shuffle_batch([image, label],
			batch_size = batch_size, capacity = 2000, min_after_dequeue = 600,
			num_threads = 20)
	else:
		image_batch, label_batch = tf.train.batch([image, label], batch_size = batch_size,
			num_threads = 20, capacity = 2000)
	if one_hot:
		label_batch = tf.one_hot(label_batch, depth = classes)
		label_batch = tf.cast(label_batch, dtype = tf.int32)
		label_batch = tf.reshape(label_batch, [batch_size, classes])

	return image_batch, label_batch


def my_batch_norm(inputs, epsilon = 1e-8):
	"""
	对输入的数据进行batch normalization

	Args:
		inputs: inputs 不是上一层的输出，而是 Wx+b，其中x才是上一层的输出
	Returns:
		inputs: 输入的数据
		batch_mean: batch内的均值
		batch_var: batch内的方差
		beta, scale: 需要训练的偏差值和权重值
	"""

	scale  = tf.Variable(tf.ones([inputs.get_shape()[-1]]), dtype = tf.float32)
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), dtype = tf.float32)
	batch_mean, batch_var = tf.nn.moments(inputs, axes = list(range(len(inputs.shape[:-1]))))

	X = tf.nn.batch_normalization(inputs, mean = batch_mean, variance = batch_var,
		offset = beta, scale = scale, variance_epsilon = epsilon)
	return X


def conv(layer_name, X, kernel_num, ksize = (3, 3), 
		 strides = [1, 1, 1, 1], padding = 'SAME', 
		 is_pretrain = True):
	"""
	执行前向的卷积运算
	"""

	in_channels = X.shape[-1]
	with tf.variable_scope(layer_name):
		W = tf.get_variable(name = 'weights',
							shape = [ksize[0], ksize[1], in_channels, kernel_num],
							dtype = tf.float32,
							initializer = tf.contrib.layers.xavier_initializer(),
							trainable = is_pretrain)
		b = tf.get_variable(name = 'biases',
							shape = [kernel_num],
							dtype = tf.float32,
							initializer = tf.constant_initializer(0.0),
							trainable = is_pretrain)
		X = tf.nn.conv2d(X, W, strides = strides, padding = padding, name = 'conv')
		X = tf.nn.bias_add(X, b, name = 'bias_add')
		return X


def pool(layer_name, X, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
		 pool_type = 'max'):
	"""
	执行前向的池化运算
	"""

	if 'max' == pool_type:
		X = tf.nn.max_pool(X, ksize, strides = strides, padding = 'SAME', 
			name = layer_name + 'max_pool')
	else:
		X = tf.nn.avg_pool(X, ksize, strides = strides, padding = 'SAME', 
			name = layer_name + 'avg_pool')
	return X


def dense(layer_name, X, num_nodes, batch_norm = True, activate_type = 'relu'):
	"""
	执行前向的全连接运算
	"""

	assert(2 == len(X.shape))
	num_features = X.shape[-1]

	with tf.variable_scope(layer_name):
		W = tf.get_variable(name = 'weights', shape = (num_features, num_nodes),
							dtype = tf.float32,
							initializer = tf.contrib.layers.xavier_initializer())
		b = tf.get_variable(name = 'biases', shape = (num_nodes), 
							dtype = tf.float32,
							initializer = tf.constant_initializer(0.0))
		X = tf.matmul(X, W)
		X = tf.nn.bias_add(X, b, name = 'Z')

		# 按要求进行batch normalization
		if batch_norm:
			X = my_batch_norm(X)

		# 非线性激活
		if 'relu' == activate_type:
			X = tf.nn.relu(X, name = 'relu_activate')
		elif 'sigmoid' == activate_type:
			X = tf.nn.sigmoid(X, name = 'sigmoid_activate')
		elif 'tanh' == activate_type:
			X = tf.nn.tanh(X, name = 'tanh_activate')
		else:
			raise ValueError

		return X


def loss_calculate(logits, labels, loss_type = 'cross_entropy'):
	"""
	计算前向传播过程中的损失函数

	Args:
		logits: 最后一层softmax层的输出
		labels: 每个样本对应的标签
		loss_type: 损失函数的类型，{'cross_entropy', 'L2'}
	"""

	with tf.name_scope('loss') as scope:
		if 'cross_entropy' == loss_type:
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
			loss = tf.reduce_mean(cross_entropy, name = 'loss')
			tf.summary.scalar(scope + '/loss', loss)
		elif 'L2' == loss_type:
			loss = tf.reduce_mean(tf.square(logits - labels), keep_dim = False, name = 'loss')
			tf.summary.scalar(scope + '/loss', loss)

		return loss


def accuracy_calculate(logits, labels):
	"""
	计算模型的准确性
	"""

	with tf.name_scope('accuracy') as scope:
		correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
		correct = tf.cast(correct, tf.float32)
		accuracy = tf.reduce_mean(correct) * 100.0
		tf.summary.scalar(scope + 'accuracy', accuracy)
		return accuracy


def optimizer(loss, global_step, optimizer_type = 'Adam', lr = 1e-4):
	"""
	设置网络的优化器

	Args:
		loss: 计算得到的损失函数
		global_step: tf.Variable类型的对象，用于记录迭代优化的次数，主要用于参数的输出和保存
		optimizer_type: str类型，设置优化器的类型，{'Adam', 'SGD'}
		lr: 设置网络的学习率
	"""

	if 'Adam' == optimizer_type:
		train_step = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss, global_step)
	elif 'SGD' == optimizer_type:
		train_step = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(loss, global_step)

	return train_step


if __name__ == '__main__':
	image_data, labels, _ = data_prepare.extract_data()
	file_path = 'dataset/train.tfrecord'
	data2tfrecord(image_data, labels, file_path)
#-*- coding: utf-8 -*-
"""
Create on 2019/5/22

@Author: xhj
"""

import numpy as np 
import model_tools
import tensorflow as tf 


def LeNet5(X, n_classes = 28, is_pretrain = True):
	"""
	实现LeNet5结构的卷积网络的搭建
	
	Args:
		X: tf.placeholder类型，网络的输入
		n_classes: int类型，网络输出的类别数
		is_pretrain: boolean类型，表示网络的参数是否需要进行训练
	"""

	with tf.name_scope('LeNet5') as scope:
		
		# 第一层卷积
		X = model_tools.conv('Conv1', X, 20, (5, 5), is_pretrain = is_pretrain)
		X = tf.nn.relu(X, name = 'Conv1/A')
		X = model_tools.pool('Conv1', X)

		# 第二层卷积
		X = model_tools.conv('Conv2', X, 50, (5, 5), is_pretrain = is_pretrain)
		X = tf.nn.relu(X, name = 'Conv2/A')
		X = model_tools.pool('Conv2', X)

		# 第三层全连接层
		X = tf.reshape(X, shape = (-1, np.prod(X.shape[1:])))
		X = model_tools.dense('fc3', X, 500)
		X = model_tools.dense('fc4', X, 500)
		X = model_tools.dense('softmax', X, n_classes)

		return X

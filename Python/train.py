#-*- coding: utf-8 -*-
"""
Create on 2019/5/22

@Author: xhj
"""

import os
import cv2
import model
import model_tools
import data_prepare
import numpy as np 
import tensorflow as tf 


def train(batch_size, epoches, shuffle, data_file = 'dataset/train.tfrecord'):
	"""
	训练模型的参数

	Args:
		batch_size: 训练数据的批次大小
		epoches: 训练模型所需要迭代的次数
		shuffle: 表示在构建mini batch时是否选哟进行顺序打乱
	"""

	log_dir = 'logs'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	
	train_log_dir = os.path.join(log_dir, 'train/')
	val_log_dir = os.path.join(log_dir, 'validation/')

	if not os.path.exists(train_log_dir):
		os.makedirs(train_log_dir)
	if not os.path.exists(val_log_dir):
		os.makedirs(val_log_dir)

	# 获取训练数据集
	train_image_batch, train_label_batch = model_tools.read_and_decode_tfrecord_files(data_file,
		batch_size)

	# 网络的调用
	X = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28, 1])
	_y = tf.placeholder(dtype = tf.int32, shape = [None, 28])
	logits = model.LeNet5(X)
	loss = model_tools.loss_calculate(logits, _y)
	accuracy = model_tools.accuracy_calculate(logits, _y)

	# 记录网络迭代优化的次数
	my_global_step = tf.Variable(0, trainable = False, name = 'global_step')
	train_step = model_tools.optimizer(loss, my_global_step, 'Adam', 1e-5)

	# 保存模型
	saver = tf.train.Saver()
	summary_op = tf.summary.merge_all()
	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		coord = tf.train.Coordinator()       # 线程协调器
		threads = tf.train.start_queue_runners(sess = sess, coord = coord)
		train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

		try:
			for step in np.arange(epoches):
				if coord.should_stop():
					break

				train_images, train_labels = sess.run([train_image_batch, train_label_batch])
				_, train_loss, train_accuracy = sess.run([train_step, loss, accuracy], 
					feed_dict = {X: train_images, _y: train_labels})

				if step % 50 == 0 or (step + 1) == epoches:
					print("Step: %d, loss: %.4f, accuracy: %.4f %%" % (step, train_loss,
						train_accuracy))
					summary_str = sess.run(summary_op, feed_dict = {X: train_images, 
						_y: train_labels})
					train_summary_writer.add_summary(summary_str, step)
		except tf.errors.OutOfRangeError:
			print("finish training")
		finally:
			coord.request_stop()

		coord.join(threads)
	

if __name__ == '__main__':
	train(256, 500, True)


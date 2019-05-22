#-*- coding: utf-8 -*-

import os
import cv2
import h5py
import numpy as np 

DATASET_DIR = '../dataset/'
TRAIN_FILE = '../dataset/train.h5'
IMG_WIDTH = 28
IMG_HEIGHT = 28

if not os.path.exists(DATASET_DIR):
	raise ValueError

label2y = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
		   'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15, 'g':16, 'h':17, 'x':18,
		   'K':19, 'Q':20, 'R':21, 'N':22, 'B':23, '+':24, '#':25, '=':26, '-':27, 'o':28}


def image_process():

	X = []
	y = []
	for data_dir in os.listdir(DATASET_DIR):
		char_dir = os.path.join(DATASET_DIR, data_dir)
		if os.path.isdir(char_dir):
			label = data_dir
			if label == 'lb':
				label = 'b'
			for img_file in os.listdir(char_dir):
				img_path = os.path.join(char_dir, img_file)
				try:
					img = cv2.imread(img_path, 2)
					height, width = img.shape[:2]
					scale = max(height, width) / IMG_WIDTH
					new_height, new_width = int(height / scale), int(width / scale)
					img = cv2.resize(img, (new_width, new_height))     # resize 至最长边为28
					if new_height % 2 != 0:
						top = (IMG_HEIGHT- new_height) // 2
						down = top + 1
						if new_width % 2 != 0:
							left = (IMG_WIDTH - new_width) // 2
							right = left + 1
						elif new_width % 2 == 0:
							left = right = (IMG_WIDTH - new_width) // 2
					else:
						top = down = (IMG_HEIGHT - new_height) // 2
						if new_width % 2 != 0:
							left = (IMG_WIDTH - new_width) // 2
							right = left + 1
						elif new_width % 2 == 0:
							left = right = (IMG_WIDTH - new_width) // 2

					img = np.pad(img, ((top, down), (left, right)), 'constant')
					X.append(img)
					y.append(label2y[label])
				except:
					print("Can't read image: %s in directory: %s ..." %(img_file, data_dir))
	return X, y


def save_to_file(X, y, label2y, filepath = TRAIN_FILE, extend_file = False):
	
	if not os.path.exists(filepath):
		with h5py.File(filepath, 'w') as train_file:
			train_file['X'] = X
			train_file['y'] = y
			# h5py的create_dataset函数直接收ascii编码的数据，而不接受utf-8编码的数据，所以需要先对str类型的数据进行转码
			# 通过ord(char)转化成ascii编码，提取时使用chr(ascii)转换回char
			train_file['label'] = list(zip([ord(x) for x in label2y.keys()], label2y.values()))   
	else:
		if extend_file:
			with h5py.File(filepath, 'r') as train_file:
				tmp_X = list(train_file['X'])
				tmp_y = list(train_file['y'])
			os.remove(filepath)
			tmp_X = tmp_X.extend(X)
			tmp_y = tmp_y.extend(y)
			with h5py.File(filepath, 'w') as train_file:
				train_file['X'] = tmp_X
				train_file['y'] = tmp_y


def extract_data(filepath = TRAIN_FILE):
	if not os.path.exists(filepath):
		raise ValueError
	with h5py.File(filepath, 'r') as train_file:
		X = list(train_file['X'])
		y = list(train_file['y'])
		label2y = list(train_file['label'])
		label2y = dict([(chr(x[0]), x[1]) for x in label2y])
		return X, y, label2y


if __name__ == '__main__':
	# save_to_file(X, y, label2y)
	X, y, label2y = extract_data()

#-*- coding: utf-8 -*-
"""
Created on 2019/5/16

@Author: xhj
"""

import os
import cv2
import numpy as np


def find_hull(img):
	"""
	输入BGR格式的图片，找到图片中表格所在位置的凸包
	"""

	if not 3 == img.ndim:
		raise ValueError
	image = img.copy()
	height, width = image.shape[:2]
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gauss_img = cv2.bilateralFilter(gray_img, 10, 50, 30)  # 双边滤波
	binary_img = cv2.adaptiveThreshold(~gauss_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
		cv2.THRESH_BINARY, 15, -2)    # 自适应阈值化

	# 对阈值图像进行膨胀，连接中断的横线，从而保证可以找到完整的表格轮廓
	kernel1 = np.ones((1, 3), np.uint8)
	binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel1)

	# 对阈值图像进行膨胀，连接中断的竖线，从而保证可以找到完整的表格轮廓
	kernel2 = np.ones((3, 1), np.uint8)
	binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel2)

	# 找到轮廓
	_, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE,
		cv2.CHAIN_APPROX_SIMPLE)
	contour_img = np.zeros((height, width, 3), np.uint8)
	contours_poly = []
	for index, contour in enumerate(contours):
		if cv2.contourArea(contour) > height * width / 4:
			# 对于符合条件的轮廓，得到逼近多边形的坐标
			contours_poly.append(cv2.approxPolyDP(contour, epsilon=5, closed=True))
	contours_poly0 = contours_poly[0]
	cv2.line(image, tuple(contours_poly0[0][0]), tuple(contours_poly0[-1][0]), (0, 0, 255), 2)
	for index in range(len(contours_poly0) - 1):
		cv2.line(image, tuple(contours_poly0[index][0]), tuple(contours_poly0[index+1][0]), (0, 0, 255), 2)

	# 从逼近多边形的点集中找到凸包
	contours_poly0 = contours_poly[0].flatten().reshape((contours_poly0.shape[0], 2))
	hull = cv2.convexHull(contours_poly0, clockwise=False, returnPoints=True)   #
	hull = hull.flatten().reshape((len(hull), 2))

	hull_selected = []
	if len(hull) >= 4:

		# print(hull)
		hull_y = sorted(hull, key = lambda x: x[1])
		# print(hull_y)
		y_mean = (hull_y[-1][1] + hull_y[0][1]) / 2   # 先求y方向，因为y方向基本是稳定的
		y_gap = hull_y[-1][1] - hull_y[0][1]
		hull_y = [x for x in hull_y if abs(x[1] - y_mean) > y_gap/3]
		# print(hull_y)

		# hull = hull_y
		hull_up = [x for x in hull_y if x[1] < y_mean]
		hull_x_up = sorted(hull_up, key = lambda x: x[0])
		x_mean_up = (hull_x_up[-1][0] + hull_x_up[0][0]) / 2
		x_gap_up = hull_x_up[-1][0] - hull_x_up[0][0]
		hull_x_up = [x for x in hull_x_up if abs(x[0] - x_mean_up) > x_gap_up/3]
		hull = hull_x_up

		hull_down = [x for x in hull_y if x[1] > y_mean]
		hull_x_down = sorted(hull_down, key = lambda x: x[0])
		# print(hull_down)
		x_mean_down = (hull_x_down[-1][0] + hull_x_down[0][0]) / 2
		x_gap_down = hull_x_down[-1][0] + hull_x_down[0][0]
		hull_x_down = [x for x in hull_x_down if abs(x[0] - x_mean_down) > x_gap_down/3]
		hull.extend(hull_x_down)

	for point in hull:
		cv2.circle(image, tuple(point), 10, (0, 255, 0), 2)

	return hull, image


def warp_table_roi(image, hull_points):
	"""
	通过得到的凸包点，对表格区域进行矫正
	"""

	if len(hull_points) != 4:
		raise ValueError

	hull_new = sorted(hull_points, key = lambda x: x[1]) # 根据y坐标进行排序
	point1, point2, point3, point4 = hull_new[:]
	# point1, point2, point3, point4 分别表示左上，右上，左下，右下
	if point1[0] > point2[0]:
		point1, point2 = point2, point1
	if point3[0] > point4[0]:
		point3, point4 = point4, point3
	point1_new = point1
	point2_new = (point2[0], point1[1])
	point3_new = (point1[0], point3[1])
	point4_new = (point2[0], point3[1])
	src = np.float32([[point1, point2, point3, point4]])
	dest = np.float32([[point1_new, point2_new,
						point3_new, point4_new]])
	M = cv2.getPerspectiveTransform(src, dest)       # 获取透视变换矩阵
	Minv = cv2.getPerspectiveTransform(dest, src)    # 获取透视变换的逆矩阵
	warped_img = cv2.warpPerspective(image, M, image.shape[1::-1],
									 flags=cv2.INTER_LINEAR)
	height, width = warped_img.shape[:2]
	# cv2.imshow("warp image", warped_img)
	row_begin = max(0, point1_new[1] - 5)
	row_end = min(height, point3_new[1] + 5)
	col_begin = max(0, point1_new[0] - 5)
	col_end = min(width, point2_new[0] + 5)
	tabel_roi = warped_img[row_begin: row_end, col_begin: col_end, :]
	cv2.imshow("warp image", tabel_roi)
	return tabel_roi


def table_extract_new(image, img_name, save_img = False, save_char_img = False):
	height, width = image.shape[:2]
	img_show = image.copy()
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gauss_img = cv2.bilateralFilter(gray_img, 5, 10, 30)

	binary_img = cv2.adaptiveThreshold(~gauss_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
		cv2.THRESH_BINARY, 15, -2)

	# 提取横线
	kernel1 = np.ones((1, 10), np.uint8)
	binary_row = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, kernel1)
	kernel12 = np.ones((2, 30), np.uint8)
	binary_row = cv2.morphologyEx(binary_row, cv2.MORPH_OPEN, kernel1)
	binary_row = cv2.morphologyEx(binary_row, cv2.MORPH_CLOSE, kernel12)
	kernel12 = np.ones((1, 30), np.uint8)
	binary_row = cv2.morphologyEx(binary_row, cv2.MORPH_DILATE, kernel12)

	# 提取竖线
	kernel2 = np.ones((10, 1), np.uint8)
	binary_col = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, kernel2)
	kernel22 = np.ones((30, 2), np.uint8)
	binary_col = cv2.morphologyEx(binary_col, cv2.MORPH_OPEN, kernel2)
	binary_col = cv2.morphologyEx(binary_col, cv2.MORPH_CLOSE, kernel22)
	kernel22 = np.ones((20, 1), np.uint8)
	binary_col = cv2.morphologyEx(binary_col, cv2.MORPH_DILATE, kernel22)

	# 过滤横线
	_, contours_row, _ = cv2.findContours(binary_row, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	new_contours_row = []
	new_contours_row_centersy = []
	_, contours_col, _ = cv2.findContours(binary_col, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	new_contours_col = []
	new_contours_col_centersx = []
	for i in range(len(contours_row)):
		rect = cv2.minAreaRect(contours_row[i])
		if max(rect[1][0], rect[1][1]) < width / 2:
			continue     # 删除掉较短的横线或者宽度较大的横线
		# new_contours_row.append(contours_row[i])
		new_contours_row_centersy.append((contours_row[i], rect[0][1]))   # 记录最小包围矩形中心的y坐标
	# binary_row = np.zeros_like(binary_img)
	# binary_row = cv2.drawContours(binary_row, new_contours_row, -1, 255, -1)
	# cv2.imshow("r", binary_row)

	# 过滤竖线
	for i in range(len(contours_col)):
		rect = cv2.minAreaRect(contours_col[i])
		if max(rect[1][0], rect[1][1]) < height / 2:
			continue     # 删除掉较短的横线或者宽度较大的横线
		# new_contours_col.append(contours_col[i])
		new_contours_col_centersx.append((contours_col[i], rect[0][0]))   # 记录最小包围矩形中心的x坐标
	# binary_col = np.zeros_like(binary_img)
	# binary_col = cv2.drawContours(binary_col, new_contours_col, -1, 255, -1)
	# cv2.imshow("c", binary_col)

	new_contours_row_centersy = sorted(new_contours_row_centersy, key = lambda x: x[1])
	new_contours_col_centersx = sorted(new_contours_col_centersx, key = lambda x: x[1])
	new_contours_row = [x[0] for x in new_contours_row_centersy]
	new_contours_col = [x[0] for x in new_contours_col_centersx]

	center_points = [[0] * len(new_contours_col) for _ in range(len(new_contours_row))]
	point_img = np.zeros_like(binary_img, dtype = np.uint8)
	for i in range(len(new_contours_row)):
		binary_row = np.zeros_like(binary_img, dtype = np.uint8)
		# 绘制第i条横线
		binary_row = cv2.drawContours(binary_row, new_contours_row, i, 255, -1)
		for j in range(len(new_contours_col)):
			binary_col = np.zeros_like(binary_col, dtype = np.uint8)
			# 绘制第j条竖线
			binary_col = cv2.drawContours(binary_col, new_contours_col, j, 255, -1)
			corner_point_img = binary_row & binary_col    # 得到第i条横线和第j条竖线的交点
			_, contours, hierarchy = cv2.findContours(corner_point_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
			point = contours[0][0][0]
			center_points[i][j] = point    # center_points 中的坐标值是按行进行保存的
			cv2.circle(img_show, tuple(point), 2, (0, 0, 255), 2)

	cv2.imshow("corner", img_show)
	print(center_points)
	sheet_roi_imgs = []
	for row in range(1, len(new_contours_row) - 1):
		for col in range(len(new_contours_col) - 1):
			if col % 3 != 0:
				row_begin = center_points[row][col][1] + 2
				col_begin = center_points[row][col][0] + 2
				row_end = center_points[row + 1][col + 1][1] - 2
				col_end = center_points[row + 1][col + 1][0] - 2
				roi = image[row_begin: row_end, col_begin: col_end, :]
				sheet_roi_imgs.append((roi, (row, col)))
				if save_img:
					img_path = '../sheet_src_imgs/' + image_name + '-' + str(row) + '-' + str(col) + '.jpg'
					cv2.imwrite(img_path, roi)
				if save_char_img:
					char_rois = word_split(roi)
					for index, v in enumerate(char_rois):
						img_path = '../character_imgs/' + image_name + '-' + str(row) + '-' + str(col) + '-'\
							+ str(index) + '.jpg'
						cv2.imwrite(img_path, v)
	cv2.imshow("roi", sheet_roi_imgs[1][0])


def word_split(image, thresh = 0.15):
	if not 3 == image.ndim:
		raise ValueError
	height, width = image.shape[:2]
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	mean_kernel = np.ones((3, 3), np.uint8) / 9
	blur_img = cv2.filter2D(gray_img, -1, mean_kernel)
	inv_img = ~blur_img

	binary_img = cv2.adaptiveThreshold(inv_img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, 
    	cv2.THRESH_BINARY, 19, -2)
	kernel_dilate = np.ones((int(height/4), 1), np.uint8)
	binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel_dilate)

	# if np.mean(binary_img) <= thresh:
	print(np.mean(binary_img))
	if np.mean(binary_img) < thresh:
		return []

	# 通过找连通域的方式来过滤掉较小的联通域
	# 同时可以通过比较最短边和最长边的比例关系来过滤掉上下表格线的轮廓
	_, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	selected_contours = []
	for contour in contours:
		if cv2.contourArea(contour) < width * height / 50:
			continue
		selected_contours.append(contour)
	contours_img = np.zeros_like(binary_img)
	cv2.drawContours(contours_img, selected_contours, -1, 1, -1)

	kernel_open = np.ones((3, 1), np.uint8)
	contours_img = cv2.morphologyEx(contours_img, cv2.MORPH_OPEN, kernel_open)
	
	row_count = np.sum(contours_img, axis = 0)
	col_count = np.sum(contours_img, axis = 1)

	col_begin = width
	col_end = 0
	for i in range(width):
	    if row_count[i] >= 2:
	        col_begin = max(0, i - 2)
	        for j in range(width):
	            if row_count[width - 1 - j] >= 2:
	                col_end = min(width, width - j + 2)
	                break
	        break

	row_begin = height
	row_end = 0
	for i in range(height):
	    if col_count[i] >= 2:
	        row_begin = max(0, i)
	        for j in range(height):
	            if col_count[height - 1 - j] >= 2:
	                row_end = min(height, height - j)
	                break
	        break
	    
	if col_begin < col_end:  # 只有满足这两个条件才进行后续步骤
	    if row_begin < row_end:
	        image = image[row_begin: row_end, col_begin: col_end, :]
	        contours_img = contours_img[row_begin: row_end, col_begin: col_end]
	        row_count = row_count[col_begin: col_end]
	        col_count = col_count[row_begin: row_end]
	        height, width = image.shape[:2]

	print(height, width)

	trough = [0]
	max_peak = max(row_count)
	print(max_peak)
	for i in range(1, len(row_count)-1):
	    # 当波谷是一条横线，找右分割边界
	    if (row_count[i] <= row_count[i - 1] and row_count[i] < row_count[i + 1])\
	        and row_count[i] < max_peak / 3:
	        if len(trough) > 0 and (i - 1 - trough[-1]) <= 3:       # 保证分割的间距要大于3个像素
	            trough[-1] = int((i - 1 + trough[-1]) / 2)
	        else:
	            trough.append(i - 1)
	            
	    # 当波谷是一条横线，找左分割边界
	    if (row_count[i] < row_count[i - 1] and row_count[i] <= row_count[i + 1])\
	        and row_count[i] < max_peak / 3:
	        if len(trough) > 0 and (i + 1 - trough[-1]) <= 3:
	            trough[-1] = int((i + 1 + trough[-1]) / 2)
	        else:
	            trough.append(i + 1)
	if (width - trough[-1] <= 3):
	    trough[-1] = width
	else:
	    trough.append(width)

	# 如果没有找到波谷，就进行均分
	if len(trough) <= 2:
	    trough.insert(1, int(width / 2))
	    
	# 如果最大的分割宽度大于第二大分割宽度的1.5倍，就对最大分割宽度进行均分
	# 记录每个分割宽度的列表
	split_width = []
	for i in range(len(trough) - 1):
	    split_width.append(trough[i + 1] - trough[i])
	# split_width.append(width - trough[-1][0])
	split_width = list(zip(range(len(split_width)), split_width))
	print("width of every characters: ", split_width)
	# 对split_width进行排序
	split_width = sorted(split_width, key = lambda x: x[1])
	print("sorted width of every characters: ", split_width)
	if split_width[-1][1] >= int(split_width[-2][1] * 1.5) and split_width[-1][1] > height:
	    trough.insert(split_width[-1][0] + 1, int(trough[split_width[-1][0]] + split_width[-1][1] / 2))

	# 对于分割宽度较小的区域，进行合并
	if len(split_width) >= 3:
	    if abs(split_width[0][0] - split_width[1][0]) == 1:
	        if split_width[0][1] + split_width[1][1] < 1.2 * split_width[2][1]:
	            trough.remove(trough[max(split_width[0][0], split_width[1][0])])

	char_rois = []
	for i in range(len(trough) - 1):
		roi = image[:, trough[i]: trough[i + 1]]
		char_rois.append(roi)

	return char_rois


if __name__ == '__main__':
	file_path = '../images/12.jpg'
	image_name = os.path.split(file_path)[-1].split('.')[0]
	print(image_name)
	image = cv2.imread(file_path)
	hull, _image = find_hull(image)
	cv2.imshow('source image', _image)
	table = warp_table_roi(image, hull)
	# print(hull)
	table_extract_new(table, image_name, False, True)
	# char_rois = word_split(image)
	# i = 0
	# for img in char_rois:
	# 	i += 1
	# 	cv2.imshow(str(i), img)
	# cv2.imshow("image", _image)
	cv2.waitKey(0)

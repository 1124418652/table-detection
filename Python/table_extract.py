#-*- coding: utf-8 -*-
"""
Created on 2019/5/15

@Author: xhj
"""

import os
import cv2
import numpy as np


file_path = '../images/6.jpg'
image = cv2.imread(file_path)
height, width = image.shape[:2]

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gauss_img = cv2.bilateralFilter(gray_img, 10, 50, 30)

kernel1 = np.ones((1, 10), np.uint8)
close_img_r = 255 - cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel1)
kernel2 = np.ones((5, 1), np.uint8)
close_img_c = 255 - cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel2)

binary_img = cv2.adaptiveThreshold(~gauss_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
	cv2.THRESH_BINARY, 15, -2)

kernel3 = np.ones((1, 3), np.uint8)
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel3)
kernel2 = np.ones((3, 1), np.uint8)
binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel2)
# binary_img2 = cv2.threshold(close_img_r, 100, 255, cv2.THRESH_BINARY)[1]


_, contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, 
		cv2.CHAIN_APPROX_SIMPLE)
# image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
contour_img = np.zeros((height, width, 3), np.uint8)
contours_poly = []
for index, contour in enumerate(contours):
	if cv2.contourArea(contour) < height * width / 4:
		continue
	# contour_img = cv2.drawContours(contour_img, contours, index, (0, 255, 0), 2)
	contours_poly.append(cv2.approxPolyDP(contour, epsilon=5, closed=True))

contours_poly0 = contours_poly[0]
cv2.line(image, tuple(contours_poly0[0][0]), tuple(contours_poly0[-1][0]), (0, 0, 255), 2)
for index in range(len(contours_poly0) - 1):
	cv2.line(image, tuple(contours_poly0[index][0]), tuple(contours_poly0[index+1][0]), (0, 0, 255), 2)
# print(cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY).dtype)
print(contours_poly[0])
print(contours_poly[0].flatten().reshape((contours_poly0.shape[0], 2)))
contours_poly0 = contours_poly[0].flatten().reshape((contours_poly0.shape[0], 2))
hull = cv2.convexHull(contours_poly0, False)
print(hull)

for point in hull:
	# print(point)
	cv2.circle(image, tuple(point[0]), 10, (0, 255, 0), 2)

cv2.imshow("gauss_img", gauss_img)
# cv2.imshow("binary_img", close_img_r)
# cv2.imshow("i", close_img_c)
cv2.imshow("image", image)

cv2.waitKey(0)
#pragma once
#ifndef __STRUCT_DEFINE_H
#define __STRUCT_DEFINE_H

#include <iostream>
#include <opencv2/core.hpp>

/**
@brief 定义保存表格中单元格图片的结构
*/
typedef struct cellImg {
	int row;                // 单元格在表格中的行索引
	int col;                // 单元格在表格中的列索引
	cv::Mat roiImg;         // 单元格图片
	cellImg(int row, int col, cv::Mat roiImg) : row(row), col(col) {
		this->roiImg = roiImg.clone();
	}
}cellImg;


/**
@brief 定义保存表格中字符图片的结构
*/
typedef struct charImg {
	int row;              // 字符所在的单元格在表格中的行索引
	int col;              // 字符所在的单元格在表格中的列索引
	int index;            // 字符在单元格中的索引
	cv::Mat roiImg;       // 字符图片
}charImg;


/**
@brief 记录表格中横向和竖线的边缘以及坐标的结构体
*/
typedef struct contoursCoord
{
	std::vector<cv::Point> contour;
	float coord;
	contoursCoord(const std::vector<cv::Point> &contour, float coord)
	{
		this->contour = contour;
		this->coord = coord;
	}
}contoursCoord;


#endif // !__STRUCT_DEFINE_H

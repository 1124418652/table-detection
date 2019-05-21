#pragma once
#ifndef __GAME_RECORD_H
#define __GAME_RECORD_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "compare.h"


/**
 @brief 定义保存表格中单元格图片的结构
 */
typedef struct cellImg {
	int row;                // 单元格在表格中的行索引
	int col;                // 单元格在表格中的列索引
	cv::Mat roiImg;         // 单元格图片
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


class GameRecord {
private:
	cv::Mat sourceImg;     // 原图片
	cv::Mat tableImg;      // 表格图片，从原图中截取的ROI
	std::vector<cellImg> cellList;     // 从该图片中提取的单元格图片
	std::vector<charImg> charList;     // 从该图片中提取的字符图片

	std::vector<cv::Point> hull;       // 保存表格的四个角点坐标

	bool _findHull();         // 找到表格图片的凸包（四个角点）
	bool _warpTableRoi();     // 使用透视变换对表格图片进行校正

	/**
	 @brief 从表格图片中提取出单元格的图片，保存于cellList中
	 */
	bool _cellExtract();
	void _wordSplit(cv::Mat cell, float thresh = 0.15);

public:

	/**
	 @brief 构造函数
	 */
	GameRecord(const cv::Mat &sourceImg);
	GameRecord(const uchar* data, int row, int col, int channel);

	/**
	 @bridf 提取表格凸包的对外接口，该函数只是调用类的私有函数_findHull()来返回结果，
	 在_findHull()中会对得到的凸包进行筛选，只返回四个角点坐标.
	 @params hullPoints: 用于记录找到的凸包的向量
	 */
	void getHull(std::vector<cv::Point2i> &hullPoints);
	void getTableImage(cv::Mat &destImg);
	void getCellImage(std::vector<cv::Mat> &cellImgList);
	void getCharImage(std::vector<charImg> &charList);
};

#endif // !__GAME_RECORD_H


// tableDetection.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include "gameRecord.h"
#include "compare.h"

using namespace std;
using namespace cv;
using namespace mypt;

int main()
{
	string imgPath = "F:/program/小弦科技实习/table-detection/images/2.jpg";
	cv::Mat image = imread(imgPath);            // 读取图片

	GameRecord grecord(image);    // 初始化对象，可以是BGR图片或者 *data, row, col, channel 作为参数
	vector<Point> hullPoints;     // 用于作为参数提取并记录表格的交点
	vector<cellImg> cellImgList;  // 用于作为参数提取并记录每个单元格的索引和图片, row, col, roiImg。row，col表示第几行第几列
	std::vector<charImg> charList;  // 用于作为参数提取并记录每个字符的索引和图片, row，col，index，roiImg。index表示为单元格中分割出的第几个字符
	cv::Mat destImg;
	grecord.getHull(hullPoints);             // 提取角点
	grecord.getTableImage(destImg);          // 提取表格图片
	grecord.getCellImage(cellImgList);       // 提取表格中的所有单元格结构
	grecord.getCharImage(charList);          // 提取表格中的所有字符结构

	cv::imshow("table image", destImg);
	cv::imshow("cell image", cellImgList.front().roiImg);
	cv::imshow("char image", charList.front().roiImg);

	//imshow("image", image);
	waitKey(0);
    
	system("pause");
	return 0;
}


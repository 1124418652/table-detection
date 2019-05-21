// tableDetection.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp>
#include "gameRecord.h"

using namespace std;
using namespace cv;

int main()
{
	string imgPath = "F:/program/小弦科技实习/table-detection/images/2.jpg";
	cv::Mat image = imread(imgPath);

	GameRecord grecord(image);
	vector<Point> hullPoints;
	vector<Mat> cellImgList;
	cv::Mat destImg;
	grecord.getHull(hullPoints);
	grecord.getTableImage(destImg);
	grecord.getCellImage(cellImgList);

	//imshow("image", image);
	waitKey(0);
    
	system("pause");
	return 0;
}


// tableDetection.cpp : �������̨Ӧ�ó������ڵ㡣
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
	string imgPath = "F:/program/С�ҿƼ�ʵϰ/table-detection/images/2.jpg";
	cv::Mat image = imread(imgPath);            // ��ȡͼƬ

	GameRecord grecord(image);    // ��ʼ�����󣬿�����BGRͼƬ���� *data, row, col, channel ��Ϊ����
	vector<Point> hullPoints;     // ������Ϊ������ȡ����¼���Ľ���
	vector<cellImg> cellImgList;  // ������Ϊ������ȡ����¼ÿ����Ԫ���������ͼƬ, row, col, roiImg��row��col��ʾ�ڼ��еڼ���
	std::vector<charImg> charList;  // ������Ϊ������ȡ����¼ÿ���ַ���������ͼƬ, row��col��index��roiImg��index��ʾΪ��Ԫ���зָ���ĵڼ����ַ�
	cv::Mat destImg;
	grecord.getHull(hullPoints);             // ��ȡ�ǵ�
	grecord.getTableImage(destImg);          // ��ȡ���ͼƬ
	grecord.getCellImage(cellImgList);       // ��ȡ����е����е�Ԫ��ṹ
	grecord.getCharImage(charList);          // ��ȡ����е������ַ��ṹ

	cv::imshow("table image", destImg);
	cv::imshow("cell image", cellImgList.front().roiImg);
	cv::imshow("char image", charList.front().roiImg);

	//imshow("image", image);
	waitKey(0);
    
	system("pause");
	return 0;
}


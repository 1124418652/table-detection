#pragma once
#ifndef __STRUCT_DEFINE_H
#define __STRUCT_DEFINE_H

#include <iostream>
#include <opencv2/core.hpp>

/**
@brief ���屣�����е�Ԫ��ͼƬ�Ľṹ
*/
typedef struct cellImg {
	int row;                // ��Ԫ���ڱ���е�������
	int col;                // ��Ԫ���ڱ���е�������
	cv::Mat roiImg;         // ��Ԫ��ͼƬ
	cellImg(int row, int col, cv::Mat roiImg) : row(row), col(col) {
		this->roiImg = roiImg.clone();
	}
}cellImg;


/**
@brief ���屣�������ַ�ͼƬ�Ľṹ
*/
typedef struct charImg {
	int row;              // �ַ����ڵĵ�Ԫ���ڱ���е�������
	int col;              // �ַ����ڵĵ�Ԫ���ڱ���е�������
	int index;            // �ַ��ڵ�Ԫ���е�����
	cv::Mat roiImg;       // �ַ�ͼƬ
}charImg;


/**
@brief ��¼����к�������ߵı�Ե�Լ�����Ľṹ��
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

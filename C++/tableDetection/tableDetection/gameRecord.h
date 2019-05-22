#pragma once
#ifndef __GAME_RECORD_H
#define __GAME_RECORD_H

#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "compare.h"
#include "myPoint.h"
#include "structDefine.h"


class GameRecord {
private:
	cv::Mat sourceImg;     // ԭͼƬ
	cv::Mat tableImg;      // ���ͼƬ����ԭͼ�н�ȡ��ROI
	std::vector<cellImg> cellList;     // �Ӹ�ͼƬ����ȡ�ĵ�Ԫ��ͼƬ
	std::vector<charImg> charList;     // �Ӹ�ͼƬ����ȡ���ַ�ͼƬ

	std::vector<cv::Point> hull;       // ��������ĸ��ǵ�����

	bool _findHull();         // �ҵ����ͼƬ��͹�����ĸ��ǵ㣩
	bool _warpTableRoi();     // ʹ��͸�ӱ任�Ա��ͼƬ����У��

	/**
	 @brief �ӱ��ͼƬ����ȡ����Ԫ���ͼƬ��������cellList��
	 */
	bool _cellExtract();
	bool _wordSplit(const cv::Mat &cell, std::vector<cv::Mat> &charRois, float thresh = 0.15);

public:

	/**
	 @brief ���캯��
	 */
	GameRecord(const cv::Mat &sourceImg);
	GameRecord(const uchar* data, int row, int col, int channel);

	/**
	 @bridf ��ȡ���͹���Ķ���ӿڣ��ú���ֻ�ǵ������˽�к���_findHull()�����ؽ����
	 ��_findHull()�л�Եõ���͹������ɸѡ��ֻ�����ĸ��ǵ�����.
	 @params hullPoints: ���ڼ�¼�ҵ���͹��������
	 */
	void getHull(std::vector<cv::Point2i> &hullPoints);
	void getTableImage(cv::Mat &destImg);
	void getCellImage(std::vector<cellImg> &cellImgList);
	void getCharImage(std::vector<charImg> &charList);
};

#endif // !__GAME_RECORD_H


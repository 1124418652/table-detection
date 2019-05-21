#include "stdafx.h"
#include "gameRecord.h"


GameRecord::GameRecord(const cv::Mat &sourceImg)
{
	if (sourceImg.empty() || 3 != sourceImg.channels())
	{
		std::cerr << "Error input!" << std::endl;
		exit(-1);
	}
	this->sourceImg = sourceImg.clone();
}


GameRecord::GameRecord(const uchar *data, int row, int col, int channel)
{
	int dataLen = row * col * channel;
	if (sizeof(data) / sizeof(uchar) != dataLen || 3 != channel || 0 == dataLen)       // ȷ��������Ŀ��ȶ�������ͨ��ͼ
	{
		std::cerr << "Error input!" << std::endl;
		exit(-1);
	}
	this->sourceImg = cv::Mat(row, col, CV_8UC3, (void *)data);
	if (this->sourceImg.empty() || 3 != this->sourceImg.channels())
	{
		std::cerr << "Error input!" << std::endl;
		exit(-1);
	}
}


bool GameRecord::_findHull()
{
	if (!this->hull.empty())
		this->hull.clear();
	cv::Mat image = this->sourceImg.clone();
	cv::Mat grayImg, gaussImg, binaryImg;         // �Ҷ�ͼ��ģ��ͼ����ֵͼ
	int height = image.rows;
	int width = image.cols;
	cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);
	cv::bilateralFilter(grayImg, gaussImg, 10, 50, 30);      // ˫���˲�
	cv::adaptiveThreshold(~gaussImg, binaryImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, 
		cv::THRESH_BINARY, 15, -2);

	// �Զ�ֵͼ��������ͣ������жϵĺ��ߣ��Ӷ���֤�����ҵ������ı������
	cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));   // 1��3�еĺ�
	cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_DILATE, kernel1);

	// �Զ�ֵͼ��������ͣ������жϵ����ߣ��Ӷ���֤�����ҵ������ı������
	cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 3));   // 3��1�еĺ�
	cv::morphologyEx(binaryImg, binaryImg, cv::MORPH_DILATE, kernel2);

	// �ҵ����ͼƬ������
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	if (contours.empty())
		return false;
	for (int i = (int)contours.size() - 1; i >= 0; --i)
	{
		if (cv::contourArea(contours[i]) < height * width / 4)
			contours.erase(contours.begin() + i);
	}
	if (contours.empty())
		return false;

	// �ҵ����ͼƬ�ıƽ������
	std::vector<cv::Point> contoursPoly;
	cv::approxPolyDP(cv::Mat(contours[0]), contoursPoly, 5, true);

	// �ҵ��ƽ�����ε�͹��
	std::vector<cv::Point> hull;
	cv::convexHull(contoursPoly, hull, false, true);
	int hullSize = (int)hull.size();

	// ��͹������ɸѡ
	if (hullSize < 4)           // ͹����ĿС��4
		return false;
	else if (hullSize == 4)     // ͹����Ŀ�պ�Ϊ4
	{
		// ���ϵ��£������ҽ��������ĸ�͹������hull������
		std::sort(hull.begin(), hull.end(), cmpY);
		if (hull[0].x < hull[1].x){
			this->hull.push_back(hull[0]);
			this->hull.push_back(hull[1]);
		}
		else{
			this->hull.push_back(hull[1]);
			this->hull.push_back(hull[0]);
		}
		if (hull[2].x < hull[3].x){
			this->hull.push_back(hull[2]);
			this->hull.push_back(hull[3]);
		}
		else{
			this->hull.push_back(hull[3]);
			this->hull.push_back(hull[2]);
		}
		return true;
	}
	else                        // ͹����Ŀ����4
	{
		std::vector<cv::Point> hully = hull;

		// ɾ��������ұ�Ե�м��͹��
		std::sort(hully.begin(), hully.end(), cmpY);
		int yGap = hully.back().y - hully.front().y;
		int yMean = (hully.back().y + hully.front().y) / 2;
		for (int i = (int)hully.size() - 1; i >= 0; --i)
		{
			if (abs(hully[i].y - yMean) <= yGap / 3)
				hully.erase(hully.begin() + i);
		}

		// ɾ��������±�Ե�м��͹��
		std::vector<cv::Point> hullxUp;         // �ϱ�Ե��͹����
		std::vector<cv::Point> hullxDown;       // �±�Ե��͹����
		for (int i = 0; i < hully.size(); ++i)
		{
			if (hully[i].y < yMean)
				hullxUp.push_back(hully[i]);
			else
				hullxDown.push_back(hully[i]);
		}
		// �ϱ�Ե͹����ɸѡ
		std::sort(hullxUp.begin(), hullxUp.end(), cmpX);
		int xGap = hullxUp.back().x - hullxUp.front().x;
		int xMean = (hullxUp.back().x + hullxUp.front().x) / 2;
		for (int i = (int)hullxUp.size() - 1; i >= 0; --i)
		{
			if (abs(hullxUp[i].x - xMean) <= xGap / 3)
				hullxUp.erase(hullxUp.begin() + i);
		}
		// �±�Ե͹����ɸѡ
		std::sort(hullxDown.begin(), hullxDown.end(), cmpX);
		int xGapDown = hullxDown.back().x - hullxDown.front().x;
		int xMeanDown = (hullxDown.back().x + hullxDown.front().x) / 2;
		for (int i = (int)hullxDown.size() - 1; i >= 0; --i)
		{
			if (abs(hullxDown[i].x - xMeanDown) <= xGapDown / 3)
				hullxDown.erase(hullxDown.begin() + i);
		}
		// ���ɸѡ�����͹�������Ǵ���4
		if ((int)hullxDown.size() + (int)hullxUp.size() > 4)
		{
			return false;
		}
		// ���ϵ��£������ҽ��������ĸ�͹������hull������
		for (int i = 0; i < hullxUp.size(); ++i)
		{
			this->hull.push_back(hullxUp[i]);
		}
		for (int j = 0; j < hullxDown.size(); ++j)
		{
			this->hull.push_back(hullxDown[j]);
		}
	}
	return true;
}


bool GameRecord::_warpTableRoi()
{
	if (this->hull.empty())             // ���hull����Ϊ�գ��ȳ��Ե���_findHull()����
		if (!this->_findHull())
			return false;
	if (4 != (int)this->hull.size())    // ������������ҵ�4��͹��������false
		return false;

	std::vector<cv::Point2f> srcPoints, destPoints;
	cv::Mat M;            // ����͸�ӱ任�ĵ�Ӧ����
	cv::Mat warpedImg;    // �����������ͼƬ

	for (int i = 0; i < 4; ++i)
		srcPoints.push_back(this->hull[i]);
	destPoints.push_back(this->hull[0]);
	destPoints.push_back(cv::Point2f(this->hull[1].x, this->hull[0].y));
	destPoints.push_back(cv::Point2f(this->hull[0].x, this->hull[2].y));
	destPoints.push_back(cv::Point2f(this->hull[1].x, this->hull[3].y));
	M = cv::getPerspectiveTransform(srcPoints, destPoints);
	cv::warpPerspective(this->sourceImg, warpedImg, M, 
		cv::Size(sourceImg.cols, sourceImg.rows), cv::INTER_LINEAR);
	int height = warpedImg.rows;
	int width = warpedImg.cols;
	int rowBegin = destPoints[0].y - 5 > 0 ? destPoints[0].y - 5 : 0;
	int rowEnd = destPoints[2].y + 5 < height ? destPoints[2].y + 5 : height;
	int colBegin = destPoints[0].x - 5 > 0 ? destPoints[0].x - 5 : 0;
	int colEnd = destPoints[1].x + 5 < width ? destPoints[1].x + 5 : width;

	this->tableImg = warpedImg(cv::Range(rowBegin, rowEnd), cv::Range(colBegin, colEnd)).clone();
	cv::imshow("table", this->tableImg);
	return true;
}


bool GameRecord::_cellExtract()
{
	if (this->tableImg.empty())            // ������Ҫȷ������ȷ��GameRecord������ȷ��tableImg����
		if (!this->_warpTableRoi())
			return false;
	if (!this->cellList.empty())
		this->cellList.clear();

	cv::Mat tableImg = this->tableImg.clone();
	cv::Mat grayImg;
	cv::Mat gaussImg;
	cv::Mat binaryImg, binaryRow, binaryCol;
	int height = tableImg.rows;
	int width = tableImg.cols;
	cv::cvtColor(tableImg, grayImg, cv::COLOR_BGR2GRAY);
	cv::bilateralFilter(grayImg, gaussImg, 5, 10, 30);
	cv::adaptiveThreshold(~gaussImg, binaryImg, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);

	// ��ȡ����еĺ���
	cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 1));
	cv::morphologyEx(binaryImg, binaryRow, cv::MORPH_ERODE, kernel1);
	cv::Mat kernel12 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 2));
	cv::morphologyEx(binaryRow, binaryRow, cv::MORPH_OPEN, kernel1);
	cv::morphologyEx(binaryRow, binaryRow, cv::MORPH_CLOSE, kernel12);
	kernel12 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 1));
	cv::morphologyEx(binaryRow, binaryRow, cv::MORPH_DILATE, kernel12);

	// ��ȡ����е�����
	cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 10));
	cv::morphologyEx(binaryImg, binaryCol, cv::MORPH_ERODE, kernel2);
	cv::Mat kernel22 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 30));
	cv::morphologyEx(binaryCol, binaryCol, cv::MORPH_OPEN, kernel2);
	cv::morphologyEx(binaryCol, binaryCol, cv::MORPH_CLOSE, kernel22);
	kernel22 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 30));
	cv::morphologyEx(binaryCol, binaryCol, cv::MORPH_DILATE, kernel22);

	// ���˺��ߣ�ͨ��������ͨ�����Сwidth��
	std::vector<std::vector<cv::Point>> contoursRow;
	std::vector<contoursCoord> contourCenterRow;
	cv::findContours(binaryRow, contoursRow, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (int i = (int)contoursRow.size() - 1; i >= 0; --i)
	{
		cv::RotatedRect rect = cv::minAreaRect(contoursRow[i]);
		int lineWidth = rect.size.width >= rect.size.height ? rect.size.width : rect.size.height;
		if (lineWidth > width / 2)
			contourCenterRow.push_back(contoursCoord(contoursRow[i], rect.center.y));
	}
	std::sort(contourCenterRow.begin(), contourCenterRow.end(), cmpCoord);
	contoursRow.clear();
	for (int i = 0; i < (int)contourCenterRow.size(); ++i)
	{
		contoursRow.push_back(contourCenterRow[i].contour);
	}
	// �������ߣ�ͨ��������ͨ�����Сheight��
	std::vector<std::vector<cv::Point>> contoursCol;
	std::vector<contoursCoord> contourCenterCol;
	cv::findContours(binaryCol, contoursCol, cv::RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (int i = (int)contoursCol.size() - 1; i >= 0; --i)
	{
		cv::RotatedRect rect = cv::minAreaRect(contoursCol[i]);
		int lineHeight = rect.size.height >= rect.size.width ? rect.size.height : rect.size.width;
		if (lineHeight > height / 2)
			contourCenterCol.push_back(contoursCoord(contoursCol[i], rect.center.x));
	}
	std::sort(contourCenterCol.begin(), contourCenterCol.end(), cmpCoord);
	contoursCol.clear();
	for (int i = 0; i < (int)contourCenterCol.size(); ++i)
	{
		contoursCol.push_back(contourCenterCol[i].contour);
	}

	// ͨ���ֱ���Ʊ��ĵ�i����ߺ͵�j�����ߣ������룬��ȡ��(i,j)���ǵ㣬�����б���
	int numRows = (int)contoursRow.size();
	int numCols = (int)contoursCol.size();
	MyPoint<int> **myPointArray;           // ����һ����ά�����¼ÿһ�к�ÿһ�еĽ�������
	myPointArray = new MyPoint<int>* [numRows];
	for (int i = 0; i < numRows; ++i)
		myPointArray[i] = new MyPoint<int> [numCols];
	for (int i = 0; i < numRows; ++i)
	{
		binaryRow = cv::Mat::zeros(height, width, CV_8UC1);
		cv::drawContours(binaryRow, contoursRow, i, 255, -1);
		for (int j = 0; j < numCols; ++j)
		{
			binaryCol = cv::Mat::zeros(height, width, CV_8UC1);
			cv::drawContours(binaryCol, contoursCol, j, 255, -1);
			cv::Mat cornerPointImg;
			cv::bitwise_and(binaryRow, binaryCol, cornerPointImg);
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(cornerPointImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
			if (!contours.empty())        // ��i�����ߺ͵�j�����ߴ��ڽ���
			{
				MyPoint<int> tmpPoint(contours.front().front().x, contours.front().front().y);
				myPointArray[i][j] = tmpPoint;
			}
		}
	}

	// ͨ���õ��Ľ������꣬��ȡcell��roi
	for (int row = 1; row < numRows - 1; ++row)
	{
		for (int col = 1; col < numCols - 1; ++col)
		{
			if (col % 3 != 0)
			{
				int rowBegin, rowEnd, colBegin, colEnd;
				if (-1 == myPointArray[row][col].y)
				{
					rowBegin = mypt::mean(myPointArray[row], numCols, false) + 2;
					MyPoint<int> *tmp = new MyPoint<int>[numRows];
					for (int i = 0; i < numRows; ++i)
						tmp[i] = myPointArray[i][col];
					colBegin = mypt::mean(tmp, numRows, true) + 2;
					delete[] tmp;
				}
				else
				{
					rowBegin = myPointArray[row][col].y + 2;
					colBegin = myPointArray[row][col].x + 2;
				}
				if (-1 == myPointArray[row + 1][col + 1].y)
				{
					rowEnd = mypt::mean(myPointArray[row + 1], numCols, false) - 2;
					MyPoint<int> *tmp = new MyPoint<int>[numRows];
					for (int i = 0; i < numRows; ++i)
						tmp[i] = myPointArray[i][col + 1];
					colEnd = mypt::mean(tmp, numCols, true) + 2;
					delete[] tmp;
				}
				else
				{
					rowEnd = myPointArray[row + 1][col + 1].y - 2;
					colEnd = myPointArray[row + 1][col + 1].x - 2;
				}
				cv::Mat roi = this->tableImg(cv::Range(rowBegin, rowEnd),
					cv::Range(colBegin, colEnd));
				this->cellList.push_back(cellImg(row, col, roi));
			}
		}
	}

	// �ͷŶ�̬������ڴ�
	for (int i = 0; i < numRows; ++i)
		delete[] myPointArray[i];
	delete [] myPointArray;
}


bool GameRecord::_wordSplit(const cv::Mat &cell, std::vector<cv::Mat> &charRois, 
	float thresh = 0.15)
{
	if (!3 == cell.channels())
		return false;
	if (!charRois.empty())
		charRois.clear();
	int height = cell.rows;
	int width = cell.cols;
	cv::Mat grayImg, blurImg, invImg;
}


void GameRecord::getHull(std::vector<cv::Point2i> &hullPoints)
{
	if (!hullPoints.empty())
		hullPoints.clear();
	if (!this->hull.empty())
		hullPoints = this->hull;
	else
	{
		if (this->_findHull())
			hullPoints = this->hull;
		else
			std::cout << "Can't find the hull!" << std::endl;
	}
}


void GameRecord::getTableImage(cv::Mat &destImg)
{
	if (this->tableImg.empty())
	{
		if (!this->_warpTableRoi()) {
			std::cout << "Can't get the table image!" << std::endl;
			return;
		}
	}
	destImg = this->tableImg;
}


void GameRecord::getCellImage(std::vector<cv::Mat> &cellImgList)
{
	if (this->_cellExtract())
	{
		std::cout << this->cellList.size() << std::endl;
		cv::imshow("img", this->cellList[6].roiImg);
	}
}
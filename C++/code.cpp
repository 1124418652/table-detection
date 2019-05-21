//定位并识别照片中的表格

#include <stdio.h>
#include <io.h>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat src = imread("src.jpg");
	Mat gray, bin;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	// 自适应滤波
	adaptiveThreshold(~gray, bin, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	Mat kernel1 = Mat::ones(1, 7, CV_8UC1);    // 求和滤波
	Mat kernel2 = Mat::ones(7, 1, CV_8UC1);
	Mat tmp1, tmp11, tmp2, tmp22;
	filter2D(bin, tmp1, -1, kernel1);     // 找横线
	filter2D(bin, tmp2, -1, kernel2);     // 找竖线
	tmp1 = tmp1 >= 6;                          // 过滤掉值比较小的点
	tmp2 = tmp2 >= 6;

	// 在横线的图中过滤掉较短的连通域
	vector<vector<Point> > contours;
	findContours(tmp1.clone(), contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (int i = contours.size() - 1; i >= 0; i--)
	{
		RotatedRect rect = minAreaRect(contours[i]);
		getRect(rect);
		if (rect.size.width < 100 || rect.size.height * 10 > rect.size.width)
		{
			contours.erase(contours.begin() + i);
		}
	}

	// 在竖线的图中过滤掉较短的连通域
	tmp11 = Mat::zeros(tmp1.size(), CV_8UC1);
	drawContours(tmp11, contours, -1, 1, -1);
	findContours(tmp2.clone(), contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for (int i = contours.size() - 1; i >= 0; i--)
	{
		RotatedRect rect = minAreaRect(contours[i]);
		getRect(rect);
		if (rect.size.width < 100 || rect.size.height * 20 > rect.size.width)
		{
			contours.erase(contours.begin() + i);
		}
	}
	tmp22 = Mat::zeros(tmp2.size(), CV_8UC1);
	drawContours(tmp22, contours, -1, 1, -1);
	Mat tmp3 = tmp11 & tmp22;
	findContours(tmp3.clone(), contours, RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(src, contours, -1, Scalar(0, 0, 255), 2);
	namedWindow("test", 0);
	imshow("test", src);
	waitKey();

	system("pause");
	return 0;
}



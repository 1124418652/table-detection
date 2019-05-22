#include "stdafx.h"
#include "compare.h"


bool cmpX(const cv::Point &p1, const cv::Point &p2)
{
	return p1.x < p2.x;
}

bool cmpY(const cv::Point &p1, const cv::Point &p2)
{
	return p1.y < p2.y;
}

bool cmpCoord(const struct contoursCoord &a1, const struct contoursCoord &a2)
{
	return a1.coord < a2.coord;
}

bool cmpVec2i(const cv::Vec2i &a1, const cv::Vec2i &a2)
{
	return a1[1] < a2[1];
}

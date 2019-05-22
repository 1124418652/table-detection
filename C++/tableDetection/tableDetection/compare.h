#pragma once

#ifndef __COMPARE_H_H
#define __COMPARE_H_H

#include <iostream>
#include <opencv2/core.hpp>
#include "gameRecord.h"
#include "structDefine.h"
#include "myPoint.h"

bool cmpX(const cv::Point &p1, const cv::Point &p2);
bool cmpY(const cv::Point &p1, const cv::Point &p2);
bool cmpCoord(const struct contoursCoord &a1, const struct contoursCoord &a2);
bool cmpVec2i(const cv::Vec2i &, const cv::Vec2i &);

namespace mypt {
	template<typename _Tp>
	_Tp mean(const MyPoint<_Tp>* pt, int num, bool x = True)
	{
		if (x) {
			_Tp sum = 0;
			int accumulate = 0;
			for (int i = 0; i < num; ++i)
			{
				if (pt[i].x != -1)
				{
					sum += pt[i].x;
					accumulate++;
				}
			}
			return accumulate == 0 ? 0 : sum / (_Tp)accumulate;
		}
		else
		{
			_Tp sum = 0;
			int accumulate = 0;
			for (int i = 0; i < num; ++i)
			{
				if (pt[i].y != -1)
				{
					sum += pt[i].y;
					accumulate++;
				}
			}
			return accumulate == 0 ? 0 : sum / (_Tp)accumulate;
		}
	}
};

#endif // 
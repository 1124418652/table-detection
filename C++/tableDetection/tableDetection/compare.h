#pragma once

#ifndef __COMPARE_H_H
#define __COMPARE_H_H

#include <iostream>
#include <opencv2/core.hpp>
#include "gameRecord.h"

bool cmpX(const cv::Point &p1, const cv::Point &p2);
bool cmpY(const cv::Point &p1, const cv::Point &p2);
bool cmpCoord(const struct contoursCoord &a1, const struct contoursCoord &a2);


#endif // 
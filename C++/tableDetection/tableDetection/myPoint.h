#pragma once
#ifndef __MY_POINT_H
#define __MY_POINT_H

#include <iostream>
#include <opencv2/core.hpp>


template<typename _Tp>
class MyPoint
{
public:
	MyPoint() : x(-1), y(-1){};
	MyPoint(_Tp x, _Tp y) :x(x), y(y) {};
	template<typename _Tp1>
	friend std::ostream& operator<<(std::ostream &, const MyPoint<_Tp1> &);
	_Tp x;
	_Tp y;
};

template<typename _Tp1>
std::ostream& operator<<(std::ostream &out, const MyPoint<_Tp1> &p)
{
	out << '[' << p.x << ',' << p.y << ']' << std::endl;
	return out;
}

#endif // !__MY_POINT_H

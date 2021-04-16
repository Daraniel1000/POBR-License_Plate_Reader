#include "opencv2/core/core.hpp"
#pragma once
class MaskFilter
{
private:
	cv::Mat& image;
	size_t w, h;
	float mask[5][5];
	const int div = 25;
	const int size = 5, move = 2;
	cv::Vec3b work(int, int) const;
public:
	MaskFilter(cv::Mat&, float[5][5]);
	cv::Mat filter() const;
};


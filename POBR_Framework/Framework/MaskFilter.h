#pragma once
#include "opencv2/core/core.hpp"
class MaskFilter
{
private:
	cv::Mat& image;
	size_t w, h;
	float mask[3][3];
	const int div = 9;
	const int size = 3, move = 1;
	cv::Vec3b work(int, int) const;
public:
	MaskFilter(cv::Mat&, float[3][3]);
	cv::Mat filter() const;
};


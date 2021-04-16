#include "opencv2/core/core.hpp"
#pragma once
class RankFilter
{
private:
	cv::Mat& image;
	size_t w, h;
	const int size, index, move;
	cv::Vec3b work(int, int) const;
	int getBrightness(const cv::Vec3b) const;
	class pix
	{
	public:
		int x, y, b;
		pix(int, int, int);
	};
public:
	RankFilter(cv::Mat&, int, int);
	cv::Mat filter() const;
};


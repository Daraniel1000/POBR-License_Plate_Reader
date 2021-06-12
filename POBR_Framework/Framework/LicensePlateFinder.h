#pragma once
#include<vector>
#include "opencv2/core/core.hpp"

class LicensePlateFinder
{
public:
	std::vector<cv::Rect> boxes;
	cv::Mat img;
	LicensePlateFinder(cv::Mat& i): img(i) {}
	void work();
	bool isPlate(std::vector<float>);
};


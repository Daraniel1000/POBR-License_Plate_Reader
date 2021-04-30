#pragma once
#include "opencv2/core/core.hpp"
#include <vector>
#include <iostream>
#include "Tools.h"
#define at at<cv::Vec3b>

class Zad2
{
	const cv::Mat& img;
	const int w, h;
	class arrow
	{
	public:
		cv::Vec3b color;
		arrow(const cv::Vec3b c) : color(c) {}
	};
	std::vector<arrow> arrows;
	void add(const cv::Vec3b c)
	{
		for each (arrow var in arrows)
		{
			if (var.color == c) return;
		}
		arrows.emplace_back(c);
	}

	void calculateValues(const cv::Vec3b color)
	{
		int S = 0, L = 0;
		Tools::getSL(S, L, img, color);
		std::cout << "Strzalka R=" << (int)color.val[2] << " G=" << (int)color.val[1]<<", ";
		std::cout << "nachylenie " << Tools::getAngle(S, img, color);
		std::cout << ", S=" << S << ", L=" << L << ", W3=" << Tools::W3(S, L) << ", M1=" << Tools::M1(img, S, color) << ", M7=" << Tools::M7(img, S, color) << std::endl;
	}
public:
	Zad2(const cv::Mat& i): img(i), w(i.cols), h(i.rows)
	{
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				if (Tools::getBrightness(img.at(cv::Point(i, j))) < 200)
				{
					add(img.at(cv::Point(i, j)));
				}
			}
		}
		//all arrows in vector, now use each colour as mask
		for each (arrow arr in arrows)
		{
			calculateValues(arr.color);
		}
	}
};


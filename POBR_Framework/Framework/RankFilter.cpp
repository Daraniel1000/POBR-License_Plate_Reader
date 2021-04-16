#include "RankFilter.h"
#include<vector>
#include<algorithm>

RankFilter::RankFilter(cv::Mat& i, int n, int idx) : image{ i }, index(idx), move(n), size(2*n + 1)
{
	w = image.cols; //indexing is typical: 512px, 0 to 511
	h = image.rows;
}

int RankFilter::getBrightness(const cv::Vec3b pix) const
{
	return pix.val[0] + pix.val[1] + pix.val[2];
}

cv::Vec3b RankFilter::work(int x, int y) const
{
	std::vector<pix> tab;
	for (int j = -move; j <= move; ++j)
	{
		for (int i = -move; i <= move; ++i)
		{
			tab.emplace_back(x+i, y+j, getBrightness(image.at<cv::Vec3b>(cv::Point(x+i, y+j))));
		}
	}
	std::sort(tab.begin(), tab.end(), [](pix a, pix b) {
		return a.b < b.b;
	});
	return image.at<cv::Vec3b>(cv::Point(tab[index].x, tab[index].y));
}

cv::Mat RankFilter::filter() const
{
	cv::Mat result;
	image.copyTo(result);
	for (int j = move; j < h - move; ++j)
	{
		for (int i = move; i < w - move; ++i)
		{
			result.at<cv::Vec3b>(cv::Point(i, j)) = work(i, j);
		}
	}
	return result;
}

RankFilter::pix::pix(int x, int y, int b)
{
	this->x = x;
	this->y = y;
	this->b = b;
}
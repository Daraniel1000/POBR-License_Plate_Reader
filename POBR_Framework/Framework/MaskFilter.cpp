#include "MaskFilter.h"

MaskFilter::MaskFilter(cv::Mat& i, float m[5][5]) : image{ i }
{
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			mask[i][j] = m[i][j];
		}
	}
	w = image.cols; //indexing is typical: 512px, 0 to 511
	h = image.rows;
}

cv::Vec3b MaskFilter::work(int x, int y) const
{
	int wr, wg, wb;
	wr = wg = wb = 0;
	cv::Vec3b px;
	for (int j = -move; j <= move; ++j)
	{
		for (int i = -move; i <= move; ++i)
		{
			px = image.at<cv::Vec3b>(cv::Point(x+i, y+j));
			wr += mask[i + move][j + move] * px.val[0];
			wg += mask[i + move][j + move] * px.val[1];
			wb += mask[i + move][j + move] * px.val[2];
		}
	}
	wr /= div; wr = std::max(0, std::min(wr, 255));
	wg /= div; wg = std::max(0, std::min(wg, 255));
	wb /= div; wb = std::max(0, std::min(wb, 255));
	return cv::Vec3b(wr, wg, wb);
}

cv::Mat MaskFilter::filter() const
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
#pragma once
#include<queue>
#include "opencv2/core/core.hpp"
#include "Tools.h"
#define at at<cv::Vec3b>
#define WHITE cv::Vec3b(255, 255, 255)
#define BLACK cv::Vec3b(0, 0, 0)

namespace ccv {

    void whitenBlues_destructive(cv::Mat& img);

    cv::Mat hough(const cv::Mat& img);

    void fill(cv::Mat& img, const cv::Mat& mask, const cv::Point lower, const cv::Point upper, const bool negative = false);

    cv::Mat negative(const cv::Mat& img);

    cv::Mat maskObject(const cv::Mat& img, const cv::Point initial, cv::Point& minBound, cv::Point& maxBound, int* count = nullptr);

    int getBrightness(const cv::Vec3b pix);

    void contrast_Destructive(cv::Mat& img, const float alpha = 3.0);

    cv::Mat contrast(const cv::Mat& img, const float alpha = 3.0);

    bool hits(const cv::Mat& img, bool** mask, const int x, const int y, bool negative = false);

    cv::Mat erode(const cv::Mat& img, bool** mask = nullptr);

    cv::Mat dilate(const cv::Mat& img, bool** mask = nullptr);

    cv::Mat open(const cv::Mat& img);

    cv::Mat close(const cv::Mat& img);

    std::vector<float> calculateValues(const cv::Mat& img);

    void calcHist(const cv::Mat& img, int hist[256]);

    void eqHist_Destructive(cv::Mat& img);

    void grayscale_Destructive(cv::Mat& img);

    cv::Mat grayscale(const cv::Mat& img);

    void thresholding_Destructive(cv::Mat& img, const uchar thresh, const bool negative = false);

    cv::Mat eqHist(cv::Mat& img);

    cv::Mat thresholding(cv::Mat& img, const uchar thresh, const bool negative = false);
}
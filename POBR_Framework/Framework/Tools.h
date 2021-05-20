#pragma once
#include "opencv2/core/core.hpp"
#include <cmath>
#define at at<cv::Vec3b>

const double pi = 3.14159265358979323846;

class Tools
{
public:

    static int getBrightness(const cv::Vec3b pix)
    {
        return (pix.val[0] + pix.val[1] + pix.val[2]) / 3;
    }

    static float W3(const int& S, const int& L)
    {
        return (L / (2.0 * sqrt(pi * S))) - 1;
    }

    static bool checkNeighbors(const int& x, const int& y, const cv::Mat& img)
    {
        if(y<img.rows-1)
            if (getBrightness(img.at(y + 1, x)) > 200) return true;
        if(y>0)
            if (getBrightness(img.at(y - 1, x)) > 200) return true;
        if (x < img.cols - 1)
            if (getBrightness(img.at(y, x + 1)) > 200) return true;
        if(x>0)
            if (getBrightness(img.at(y, x - 1)) > 200) return true;
        return false;
    }

    static float mpq(const int p, const int q, const cv::Mat& img, const cv::Vec3b color)
    {
        const int w = img.cols; //indexing is typical: 512px, 0 to 511
        const int h = img.rows;
        float m = 0;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                if (img.at(cv::Point(i, j)) == color)
                {
                    m += pow(i, p) * pow(j, q);
                }
            }
        }
        return m;
    }

    static float Mpq(const int p, const int q, const cv::Mat& img, const int& S, const cv::Vec3b color)
    {
        const float ic = mpq(1, 0, img, color) / S;
        const float jc = mpq(0, 1, img, color) / S;
        float M = 0;
        const int w = img.cols; //indexing is typical: 512px, 0 to 511
        const int h = img.rows;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                if (img.at(cv::Point(i, j)) == color)
                {
                    M += pow(i - ic, p) * pow(j - jc, q);
                }
            }
        }
        return M;
    }

    static float Npq(const int p, const int q, const cv::Mat& img, const int& S, const cv::Vec3b color)
    {
        return Mpq(p, q, img, S, color) / pow(S, ((p + q) / 2) + 1);
    }

    static float M1(const cv::Mat& img, const int& S, const cv::Vec3b color = cv::Vec3b(0, 0, 0))
    {
        return Npq(2, 0, img, S, color) + Npq(0, 2, img, S, color);
    }

    static float M7(const cv::Mat& img, const int& S, const cv::Vec3b color = cv::Vec3b(0, 0, 0))
    {
        return Npq(2, 0, img, S, color) * Npq(0, 2, img, S, color) - pow(Npq(1, 1, img, S, color), 2);
    }

    static void getSL(int& S, int& L, const cv::Mat& img, const cv::Vec3b color = cv::Vec3b(0,0,0))
    {
        cv::Mat original;
        img.copyTo(original);
        const int w = img.cols; //indexing is typical: 512px, 0 to 511
        const int h = img.rows;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                if (img.at(cv::Point(i, j)) == color)
                {
                    ++S;
                    if (checkNeighbors(i, j, original))
                    {
                        ++L;
                    }
                }
            }
        }
    }

    static float getAngle(const int& S, const cv::Mat& img, const cv::Vec3b color)
    {
        const int w = img.cols; //indexing is typical: 512px, 0 to 511
        const int h = img.rows;
        const float ic = mpq(1, 0, img, color) / S;
        const float jc = mpq(0, 1, img, color) / S;
        int x_max=0, x_min=w;
        int y_max=0, y_min=h;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                if (img.at(cv::Point(i, j)) == color)
                {
                    if (i > x_max)x_max = i;
                    if (i < x_min)x_min = i;
                    if (j > y_max)y_max = j;
                    if (j < y_min)y_min = j;
                }
            }
        }
        const float xc = x_min + ((x_max - x_min) / 2);
        const float yc = y_min + ((y_max - y_min) / 2);
        return AngleBetweenPoints(xc, yc, ic, jc);
    }

    static float AngleBetweenPoints(const float x1, const float y1, const float x2, const float y2)
    {
        return atan2f(y1-y2, x1-x2) * 180 / pi; 
    }
};


#pragma once
#include "opencv2/core/core.hpp"
#include "Tools.h"
#define at at<cv::Vec3b>
#define WHITE cv::Vec3b(255, 255, 255)
#define BLACK  cv::Vec3b(0, 0, 0)

namespace ccv {


    int getBrightness(const cv::Vec3b pix)
    {
        return (pix.val[0] + pix.val[1] + pix.val[2]) / 3;
    }

    void contrast_Destructive(cv::Mat& img, const float alpha = 3.0)
    {
        const int w = img.cols;
        const int h = img.rows;
        float b;
        uchar x;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                b = getBrightness(img.at(j, i));
                b = (b < 128) ? b / alpha : b * alpha;
                x = cv::saturate_cast<uchar>(b);
                img.at(j, i) = cv::Vec3b(x, x, x);
            }
        }
    }

    cv::Mat contrast(const cv::Mat& img, const float alpha = 3.0)
    {
        cv::Mat copy;
        img.copyTo(copy);
        contrast_Destructive(copy, alpha);
        return copy;
    }

    bool hits(const cv::Mat& img, bool** mask, const int x, const int y, bool negative = false)
    {
        const int w = img.cols;
        const int h = img.rows;
        for (int j = -1; j < 2; ++j)
        {
            for (int i = -1; i < 2; ++i)
            {
                if (x + i<0 || x + i>=w || y + j<0 || y + j>=h) continue;
                if (mask[j + 1][i + 1] && img.at(cv::Point(x + i, y + j)) == (negative?WHITE:BLACK)) return true;
            }
        }
        return false;
    }

    cv::Mat erode(const cv::Mat& img, bool** mask = nullptr)
    {
        if (mask == nullptr)
        {
            mask = new bool* [3];
            for (int i = 0; i < 3; ++i)
            {
                mask[i] = new bool[3];
                for (int j = 0; j < 3; ++j)
                    mask[i][j] = 1;
            }
        }
        const int w = img.cols;
        const int h = img.rows;
        cv::Mat copy = cv::Mat::zeros(cv::Size(w, h), CV_8UC3);
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                copy.at(cv::Point(i, j)) = hits(img, mask, i, j, true) ? WHITE : BLACK;
            }
        }
        return copy;

    }

    cv::Mat dilate(const cv::Mat& img, bool** mask = nullptr)
    {
        //const cv::Vec3b WHITE = cv::Vec3b(255, 255, 255);
        //const cv::Vec3b BLACK = cv::Vec3b(0, 0, 0);
        if (mask == nullptr)
        {
            mask = new bool*[3];
            for (int i = 0; i < 3; ++i)
            {
                mask[i] = new bool[3];
                for (int j = 0; j < 3; ++j)
                    mask[i][j] = 1;
            }
        }
        const int w = img.cols;
        const int h = img.rows;
        cv::Mat copy = cv::Mat::zeros(cv::Size(w, h), CV_8UC3);
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                copy.at(cv::Point(i, j)) = hits(img, mask, i, j) ? BLACK : WHITE;
            }
        }
        return copy;
    }

    cv::Mat open(const cv::Mat& img)
    {
        return dilate(erode(img));
    }

    cv::Mat close(const cv::Mat& img)
    {
        return erode(dilate(img));
    }

    void calculateValues(const cv::Mat& img, float& W3, float& M1, float& M7)
    {
        int S = 0, L = 0;
        Tools::getSL(S, L, img);
        W3 = Tools::W3(S, L);
        M1 = Tools::M1(img, S);
        M7 = Tools::M7(img, S);
    }

    void calcHist(const cv::Mat& img, int hist[256])   //grayscale
    {
        for (int i = 0; i < 256; ++i)
        {
            hist[i] = 0;
        }
        const int w = img.cols;
        const int h = img.rows;
        uchar v;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                hist[img.at(cv::Point(i, j)).val[0]]++;
            }
        }
    }

    void eqHist_Destructive(cv::Mat& img)
    {
        int hist[256], LUT[256];
        calcHist(img, hist);        
        for (int i = 0; i < 256; ++i)
        {
            LUT[i] = 0;
        }
        const int w = img.cols;
        const int h = img.rows;
        int N = w * h;
        int sump = 0;
        uchar v;
        for (int i = 0; i < 256; ++i)
        {
            sump += hist[i];
            LUT[i] = sump * 255 / N;
        }
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                v = img.at(cv::Point(i, j)).val[0];
                img.at(cv::Point(i, j)) = cv::Vec3b(LUT[v], LUT[v], LUT[v]);
            }
        }
    }

    void grayscale_Destructive(cv::Mat& img)
    {
        const int w = img.cols; //indexing is typical: 512px, 0 to 511
        const int h = img.rows;
        uchar v;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                v = getBrightness(img.at(cv::Point(i, j)));
                img.at(cv::Point(i, j)) = cv::Vec3b(v, v, v);
            }
        }
    }

    void thresholding_Destructive(cv::Mat& img, const uchar thresh, const bool negative = false)
    {
        //const cv::Vec3b WHITE = cv::Vec3b(255, 255, 255);
        //const cv::Vec3b BLACK = cv::Vec3b(0, 0, 0);
        const int w = img.cols; //indexing is typical: 512px, 0 to 511
        const int h = img.rows;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                if (getBrightness(img.at(cv::Point(i, j))) > thresh)
                {
                    img.at(cv::Point(i, j)) = negative ? BLACK : WHITE;
                }
                else
                {
                    img.at(cv::Point(i, j)) = negative ? WHITE : BLACK;
                }
            }
        }
    }

    cv::Mat eqHist(cv::Mat& img)
    {
        cv::Mat copy;
        img.copyTo(copy);
        eqHist_Destructive(copy);
        return copy;
    }

    cv::Mat thresholding(cv::Mat& img, const uchar thresh, const bool negative = false)
    {
        cv::Mat copy;
        img.copyTo(copy);
        thresholding_Destructive(copy, thresh, negative);
        return copy;
    }

    void RGBtoHSV_Destructive(cv::Mat& img) //H - 0, S - 1, V - 2
    {                                       //H 0 to 180
        const int w = img.cols; //indexing is typical: 512px, 0 to 511
        const int h = img.rows;
        cv::Vec3f pix;
        int r, g, b, cmax, cmin, delta, temp;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                pix = img.at(cv::Point(i, j));
                r = pix.val[2];
                g = pix.val[1];
                b = pix.val[0];
                cmax = std::max(r, std::max(g, b));
                cmin = std::min(r, std::min(g, b));
                delta = cmax - cmin;
                pix[2] = cmax;
                if (cmax == 0)
                {
                    pix[1] = 0;
                }
                else
                {
                    pix[1] = (delta / cmax) * 255;
                }
                if (delta == 0)
                {
                    pix[0] = 0;
                }
                else
                {
                    if (cmax == r)
                    {
                        temp = (g - b) / delta;
                        temp = temp % 6;
                    }
                    else if (cmax == g)
                    {
                        temp = (b - r) / delta;
                        temp += 2;
                    }
                    else
                    {
                        temp = (r - g) / delta;
                        temp += 4;
                    }
                    pix[0] = temp;
                }
                img.at(cv::Point(i, j)) = pix;
            }
        }
    }
}
#pragma once
#include "CvFuncs.h"
#include<queue>
#include "Tools.h"
#include<math.h>
#define at at<cv::Vec3b>
#define WHITE cv::Vec3b(255, 255, 255)
#define BLACK cv::Vec3b(0, 0, 0)


    cv::Mat ccv::negative(const cv::Mat& img)
    {
        cv::Mat copy = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC3);
        const int w = img.cols;
        const int h = img.rows;
        for (int j = 0; j < h; ++j)
        {
            for (int i = 0; i < w; ++i)
            {
                copy.at(j, i) = img.at(j, i)[0] == 0 ? WHITE : BLACK;
            }
        }
        return copy;
    }

    cv::Vec3b increment(cv::Vec3b& original)
    {
        cv::Vec3b copy = original;
        copy[0]++;
        copy[1]++;
        copy[2]++;
        return copy;
    }

    cv::Mat ccv::hough(const cv::Mat& img)
    {
        const int w = img.cols;
        const int h = img.rows;

        float theta_max = 1.0 * pi;
        float theta_min = 0.0;

        float r_min = 0.0;
        float r_max = std::sqrt(w*w + h*h);

        int r_dim = 200;
        int theta_dim = 300;

        float theta, r, ir;

        cv::Mat hough_space = cv::Mat::zeros(cv::Size(r_dim, theta_dim), CV_8UC3);
        for (int x = 0; x < w; ++x)
        {
            for (int y = 0; y < h; ++y)
            {
                if (img.at(y, x)[0] == 0)
                    continue;
                for (int itheta = 0; itheta < theta_dim; ++itheta)
                {
                    theta = 1.0 * itheta * theta_max / theta_dim;
                    r = x * std::cos(theta) + y * std::sin(theta);
                    ir = r_dim * (1.0 * r) / r_max;
                    if (ir < 0) ir = r_dim - 1;
                    hough_space.at(cv::Point(ir, itheta)) = increment(hough_space.at(cv::Point(ir, itheta)));
                }
            }
        }
        return hough_space;
    }

    void ccv::blacken(cv::Mat& img, const cv::Mat& mask, const cv::Point lower, const cv::Point upper)
    {
        assert(mask.cols == img.cols && mask.rows == img.rows);
        const int w = img.cols;
        const int h = img.rows;
        for (int j = lower.y; j <= upper.y; ++j)
        {
            for (int i = lower.x; i <= upper.x; ++i)
            {
                if (mask.at(j, i) == WHITE) img.at(cv::Point(i, j)) = BLACK;
            }
        }
    }

    cv::Mat ccv::maskObject(const cv::Mat& img, const cv::Point initial, cv::Point& minBound, cv::Point& maxBound, int* count)
    {
        std::queue<cv::Point> toSearch;
        toSearch.push(initial);
        cv::Mat mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC3);
        cv::Point p, temp;
        int i, j;
        int maxX = 0, minX = img.cols, maxY = 0, minY = img.rows, pixels=0;
        while (!toSearch.empty())
        {
            p = toSearch.front();
            toSearch.pop();
            if (mask.at(p)[0] > 0) continue;
            if (p.x > maxX) maxX = p.x;
            if (p.y > maxY) maxY = p.y;
            if (p.x < minX) minX = p.x;
            if (p.y < minY) minY = p.y;
            mask.at(p) = WHITE;
            ++pixels;
            for (j = -1; j < 2; ++j)
            {
                for (i = -1; i < 2; ++i)
                {
                    temp = cv::Point(p.x + i, p.y + j);
                    if (temp.x < 0 || temp.x >= img.cols) continue;
                    if (temp.y < 0 || temp.y >= img.rows) continue;
                    if (mask.at(temp)[0] > 0) continue;
                    if (img.at(temp) == BLACK)
                    {
                        toSearch.push(temp);
                    }
                }
            }
        }
        minBound = cv::Point(minX, minY);
        maxBound = cv::Point(maxX, maxY);
        if (count != nullptr) *count = pixels;
        return mask;
    }

    int ccv::getBrightness(const cv::Vec3b pix)
    {
        return (pix.val[0] + pix.val[1] + pix.val[2]) / 3;
    }

    void ccv::contrast_Destructive(cv::Mat& img, const float alpha)
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

    cv::Mat ccv::contrast(const cv::Mat& img, const float alpha)
    {
        cv::Mat copy;
        img.copyTo(copy);
        contrast_Destructive(copy, alpha);
        return copy;
    }

    bool ccv::hits(const cv::Mat& img, bool** mask, const int x, const int y, bool negative)
    {
        const int w = img.cols;
        const int h = img.rows;
        for (int j = -1; j < 2; ++j)
        {
            for (int i = -1; i < 2; ++i)
            {
                if (x + i < 0 || x + i >= w || y + j < 0 || y + j >= h) continue;
                if (mask[j + 1][i + 1] && img.at(cv::Point(x + i, y + j)) == (negative ? WHITE : BLACK)) return true;
            }
        }
        return false;
    }

    cv::Mat ccv::erode(const cv::Mat& img, bool** mask)
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

    cv::Mat ccv::dilate(const cv::Mat& img, bool** mask)
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
                copy.at(cv::Point(i, j)) = hits(img, mask, i, j) ? BLACK : WHITE;
            }
        }
        return copy;
    }

    cv::Mat ccv::open(const cv::Mat& img)
    {
        return dilate(dilate(erode(erode(img))));
    }

    cv::Mat ccv::close(const cv::Mat& img)
    {
        return erode(dilate(img));
    }

    std::vector<float> ccv::calculateValues(const cv::Mat& img)
    {
        int S = 0, L = 0;
        Tools::getSL(S, L, img);
        std::vector<float> vals;
        vals.push_back(Tools::W3(S, L));
        vals.push_back(Tools::M1(img, S));
        vals.push_back(Tools::M2(img, S));
        //vals.push_back(Tools::M3(img, S));
        //vals.push_back(Tools::M4(img, S));
        //vals.push_back(Tools::M5(img, S));
        //vals.push_back(Tools::M6(img, S));
        vals.push_back(Tools::M7(img, S));
        //vals.push_back(Tools::M8(img, S));
        //vals.push_back(Tools::M9(img, S));
        //vals.push_back(Tools::M10(img, S));
        return vals;
    }

    void ccv::calcHist(const cv::Mat& img, int hist[256])   //grayscale
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

    void ccv::eqHist_Destructive(cv::Mat& img)
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

    void ccv::grayscale_Destructive(cv::Mat& img)
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

    cv::Mat ccv::grayscale(const cv::Mat& img)
    {
        cv::Mat copy;
        img.copyTo(copy);
        grayscale_Destructive(copy);
        return copy;
    }

    void ccv::thresholding_Destructive(cv::Mat& img, const uchar thresh, const bool negative)
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

    cv::Mat ccv::eqHist(cv::Mat& img)
    {
        cv::Mat copy;
        img.copyTo(copy);
        eqHist_Destructive(copy);
        return copy;
    }

    cv::Mat ccv::thresholding(cv::Mat& img, const uchar thresh, const bool negative)
    {
        cv::Mat copy;
        img.copyTo(copy);
        thresholding_Destructive(copy, thresh, negative);
        return copy;
    }

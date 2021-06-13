#include "LicensePlateFinder.h"
#include "CvFuncs.h"

void showAndWait(const cv::Mat& img, const std::string s = "");

void LicensePlateFinder::work()
{
    ccv::grayscale_Destructive(img);
    ccv::contrast_Destructive(img, 1.5F);
    ccv::thresholding_Destructive(img, 80);
    const int w = img.cols;
    const int h = img.rows;
    std::vector<float> vals;
    int count;
    cv::Point upper, lower;
    cv::Mat objMask;
    img = ccv::erode(ccv::erode(ccv::dilate(ccv::dilate(img))));
    for (int j = 0; j < h; ++j)
    {
        for (int i = 0; i < w; ++i)
        {
            if (img.at(cv::Point(i, j)) == cv::Vec3b(255,255,255))
            {
                objMask = ccv::maskObject(ccv::negative(img), cv::Point(i, j), lower, upper, &count);
                cv::Mat cropped(objMask, cv::Rect(lower, upper));
                if (count > 500)
                {
                    vals = ccv::calculateValues(ccv::negative(cropped));
                    if (isPlate(vals))
                    {
                        boxes.push_back(cv::Rect(lower, upper));
                    }
                }
                ccv::fill(img, objMask, lower, upper);
            }
        }
    }
}

bool LicensePlateFinder::isPlate(std::vector<float> vals)
{
    float maxvals[4] = {5, 0.7, 0.4, 0.019};
    float minvals[4] = {3, 0.4, 0.14, 0.012};
    for (int i = 0; i < vals.size(); ++i)
    {
        if (vals[i] < minvals[i] || vals[i] > maxvals[i]) return false;
    }
    return true;
}
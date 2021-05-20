#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include<fstream>
#include "CvFuncs.h"
//#define at at<cv::Vec3b>

void work() {
    std::string s;
    std::cout << "filename: ";
    std::cin >> s;
    cv::Mat img = cv::imread("wzorce/" + s + ".jpg");
    if (img.empty()) return;
    ccv::grayscale_Destructive(img);
    ccv::contrast_Destructive(img, 2.0F);
    cv::Mat copy = ccv::eqHist(img);
    ccv::thresholding_Destructive(copy, 100);
    copy = ccv::open(ccv::close(copy));
    cv::imshow("open(close)", copy);
    //cv::imshow("Close(open)", ccv::close(ccv::open(copy)));
    float W3, M1, M7;
    ccv::calculateValues(copy, W3, M1, M7); //M1 and M7 weighted towards center of object already
    //further TODO - script calculation and saving values of all template pictures
    //even further TODO - script checking for closest match in template values
    //even even further TODO - script searching for object masks in original picture cropped to board
    std::cout << s << ": W3=" << W3 << ", M1=" << M1 << ", M7=" << M7 << std::endl;
    cv::waitKey();
    cv::destroyAllWindows();
}

int main(int, char *[]) {
    while (true)
    {
        work();
    }
    /*std::ifstream fin(s + ".txt");
    int l, t, r, b;
    fin >> l >> t >> r >> b;
    cv::Rect rect(cv::Point(l,t), cv::Point(r,b));
    ccv::eqHist_Destructive(img);
    int W3, M1, M7;
    ccv::calculateValues(img, W3, M1, M7);
    std::cout << s << ": W3=" << W3 << ", M1=" << M1 << ", M7=" << M7 << std::endl;*/
    return 0;
}

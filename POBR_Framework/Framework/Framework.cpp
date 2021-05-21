#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "CvFuncs.h"
//#define at at<cv::Vec3b>

const std::vector<std::string> templates({ "0", "1", "2", "4", "5", "6", "7", "8", "9", "d", "e", "f", "g", "j", "k", "p", "r", "s", "v", "w", "x", "y" });

void prepTemplates() {
    std::ofstream fout("wzorce/log.txt");
    for each (std::string s in templates)
    {
        cv::Mat img = cv::imread("wzorce/" + s + ".jpg");
        std::ofstream file("wzorce/" + s + ".txt");
        if (!file.is_open() || img.empty()) throw std::runtime_error("Required template file is missing");
        ccv::grayscale_Destructive(img);
        ccv::contrast_Destructive(img, 2.0F);
        ccv::eqHist_Destructive(img);
        ccv::thresholding_Destructive(img, 100);
        img = ccv::open(ccv::close(img));
        float W3, M1, M7;
        ccv::calculateValues(img, W3, M1, M7); //M1 and M7 weighted towards center of object already
        fout << s << ": W3=" << W3 << ", M1=" << M1 << ", M7=" << M7 << std::endl;
        file << W3 << std::endl << M1 << std::endl << M7;
        file.close();
        cv::imshow("", img);
        cv::waitKey();
        cv::destroyAllWindows();
    }
    fout.close();
}

std::string compare(const cv::Mat& img)
{
    cv::Mat copy = ccv::negative(img);
    float W3, M1, M7;
    ccv::calculateValues(copy, W3, M1, M7);
    float W3t, M1t, M7t;
    int mind = 0, i = 0;
    M1 *= 10; M7 *= 100;
    std::vector<float> distances;
    for each (std::string s in templates)
    {
        std::ifstream fin("wzorce/" + s + ".txt");
        if (!fin.is_open()) throw std::runtime_error("Required template file is missing");
        fin >> W3t >> M1t >> M7t;
        M1t *= 10; M7t *= 100;
        distances.push_back(std::abs(W3 - W3t) + std::abs(M1 - M1t) + std::abs(M7 - M7t));
        if (distances[i] < distances[mind]) mind = i;
        ++i;
    }
    return templates[mind];
}

void work(std::string s)
{
    std::ifstream fin("projekt/" + s + ".txt");
    int l, t, r, b;
    fin >> l >> t >> r >> b;
    cv::Rect rect(cv::Point(l, t), cv::Point(r, b));
    cv::Mat img = cv::imread("projekt/" + s + ".jpg"), obj; 
    cv::Mat copy(img, rect);
    cv::Mat cropped;
    cv::resize(copy, cropped, cv::Size(256*copy.cols/copy.rows, 256));
    ccv::grayscale_Destructive(cropped);
    ccv::contrast_Destructive(cropped, 2.0F);
    cv::imshow("", cropped);
    cv::waitKey();
    cv::destroyAllWindows();
    ccv::eqHist_Destructive(cropped);
    cv::imshow("", cropped);
    cv::waitKey();
    cv::destroyAllWindows();
    ccv::thresholding_Destructive(cropped, 100);
    cropped = ccv::open(ccv::close(cropped));
    cv::imshow("", cropped);
    cv::waitKey();
    cv::destroyAllWindows();
    cv::Point p1, p2, start;
    start = cv::Point(0, cropped.rows / 2);
    std::string code = "";
    while (start.x < cropped.cols - 1)
    {
        while (cropped.at(start)[0] == 0 && start.x < cropped.cols - 1)
            start.x++;
        while (cropped.at(start)[0] > 0 && start.x < cropped.cols - 1)
            start.x++;
        if (start.x == cropped.cols) break;
        obj = ccv::maskObject(cropped, start, p1, p2);
        code += compare(cv::Mat(obj, cv::Rect(p1, p2)));
        cv::imshow("", cv::Mat(obj, cv::Rect(p1, p2)));
        std::cout << code[code.length() - 1];
        cv::waitKey();
        cv::destroyAllWindows();
        start.y = p1.y + (p2.y - p1.y) / 4;
        start.x = p2.x;
        if (code.length() >= 7) break;
    }
    std::cout << "\nCode recovered: " << code << std::endl;
    if (start.x == cropped.cols) std::cout << "Did not hit every object" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Prepare templates? y/n\n";
    std::string s;
    std::cin >> s;
    bool b = false;
    if (s == "y" || s == "Y") b = true;
    else
    {
        if (s != "n" && s != "N")
        {
            std::cout << "Input not recognized\n";
            return 0;
        }
    }
    if (b)//argc == 1)
    {
        prepTemplates();
    }
    std::cout << "Enter 0 to exit the program\n";
    while (true)
    {
        std::string s;
        std::cout << "Image id: ";
        std::cin >> s;
        if (s == "0") break;
        work(s);
    }
    return 0;
}

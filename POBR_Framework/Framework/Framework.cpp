#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "CvFuncs.h"
#include "LicensePlateFinder.h"
//#define at at<cv::Vec3b>

const std::vector<std::string> templates({ "0", "1", "2", "4", "5", "6", "7", "8", "9", "d", "e", "f", "g", "j", "k", "p", "r", "s", "v", "w", "x", "y" });

void showAndWait(const cv::Mat& img, const std::string s="")
{
    cv::imshow(s, img);
    cv::waitKey();
    cv::destroyAllWindows();
}

void prepTemplates() {
    std::ofstream fout("wzorce/_log.txt");
    for each (std::string s in templates)
    {
        std::cout << "Preparing template for " << s << std::endl;
        cv::Mat img = cv::imread("wzorce/" + s + ".jpg");
        std::ofstream file("wzorce/" + s + ".txt");
        if (!file.is_open() || img.empty()) throw std::runtime_error("Required template file is missing");
        ccv::grayscale_Destructive(img);
        ccv::contrast_Destructive(img, 2.0F);
        ccv::eqHist_Destructive(img);
        ccv::thresholding_Destructive(img, 100);
        img = ccv::open(ccv::close(img));
        cv::Point start = cv::Point(0, img.rows / 2), p1, p2;
        while (img.at(start)[0] == 0) start.x++;
        while (img.at(start)[0] > 0) start.x++;
        cv::Mat obj = ccv::maskObject(img, start, p1, p2);
        std::vector<float> vals = ccv::calculateValues(ccv::negative(cv::Mat(obj, cv::Rect(p1, p2))));
        fout << s << ":";
        int i = 1;
        for each (float var in vals)
        {
            fout << " M" << i++ << "=" << var;
            file << var << std::endl;
        }
        fout << "\n";
        file.close();
        //showAndWait(img);
    }
    fout.close();
}

std::string compare(const cv::Mat& img)
{
    //maybe normalize the values?
    cv::Mat copy = ccv::negative(img);
    std::vector<float> vals = ccv::calculateValues(copy);
    int mind = 0, i = 0;
    std::vector<std::vector<float>> all_values;
    const int factors[4] = {1, 10, 10, 100 };
    for each (std::string s in templates)
    {
        std::ifstream fin("wzorce/" + s + ".txt");
        if (!fin.is_open()) throw std::runtime_error("Required template file is missing");
        float Nt, dist = 0;
        int j = 0;
        std::vector<float> valst;
        for each (float var in vals)
        {
            fin >> Nt;
            valst.push_back(Nt);
            //dist += std::abs(var - Nt) * factors[j++];
        }
        all_values.push_back(valst);
        //if (distances[i] < distances[mind]) mind = i;
        ++i;
        fin.close();
    }
    //now for every val scale to [0, 10]
    for (int i = 0; i < vals.size(); ++i)
    {
        std::vector<float> thisN;
        int mini=0, maxi=0, j=0;
        for each (std::vector<float> vars in all_values)
        {
            thisN.push_back(vars[i]);
            if (thisN[j] < thisN[mini]) 
                mini = j;
            if (thisN[j] > thisN[maxi])
                maxi = j;
            ++j;
        }
        float offset, multiplier;
        offset = -thisN[mini];
        multiplier = 10 / (thisN[maxi] + offset);
        for (int k=0; k < all_values.size(); ++k)
        {
            all_values[k][i] = std::abs(((all_values[k][i] + offset) - (vals[i] + offset))* multiplier);
        }
    }
    i = 0;
    std::vector<float> distances;
    for each (std::vector<float> vars in all_values)
    {
        float dist = 0;
        for each (float v in vars)
        {
            dist += v;
        }
        distances.push_back(dist);
    }
    for each (float dist in distances)
    {
        if (dist < distances[mind]) 
            mind = i;
        ++i;
    }
    return templates[mind];
}

void work(std::string s, cv::Rect rect)
{
    cv::Mat img = cv::imread("projekt/" + s + ".jpg"), obj;
    if (img.empty())
    {
        std::cout << "Files not found\n";
        return;
    }
    cv::resize(img, obj, cv::Size(1280, 720));
    cv::Mat copy(obj, rect);
    cv::Mat cropped;
    cv::resize(copy, cropped, cv::Size(256*copy.cols/copy.rows, 256));
    ccv::grayscale_Destructive(cropped);
    ccv::contrast_Destructive(cropped, 2.0F);
    ccv::eqHist_Destructive(cropped);
    ccv::thresholding_Destructive(cropped, 100);
    cropped = ccv::open(ccv::close(cropped));
    //showAndWait(cropped);
    cv::Point p1, p2, start;
    start = cv::Point(0, cropped.rows / 2);
    std::string code = "";
    while (start.x < cropped.cols - 1)
    {
        while (cropped.at(start)[0] == 0 && start.x < cropped.cols - 1)
            start.x++;
        while (cropped.at(start)[0] > 0 && start.x < cropped.cols - 1)
            start.x++;
        if (start.x >= cropped.cols-1) break;
        p1 = start;
        while (cropped.at(p1)[0] == 0 && p1.x < cropped.cols - 1)
            p1.x++;
        if (p1.x >= cropped.cols - 1) break;
        obj = ccv::maskObject(cropped, start, p1, p2);
        code += compare(cv::Mat(obj, cv::Rect(p1, p2)));
        //cv::imshow("", cv::Mat(obj, cv::Rect(p1, p2)));
        //std::cout << code[code.length() - 1];
        //cv::waitKey();
        cv::destroyAllWindows();
        start.y = p1.y + (p2.y - p1.y) / 4;
        start.x = p2.x;
        //if (code.length() >= 7) break;
    }
    std::cout << "Code recovered: " << code << "\n\n";
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
    if (b)
    {
        prepTemplates();
    }
    while (true)
    {
        std::cout << "Enter 0 to exit the program\n";
        std::string s;
        std::cout << "Image id: ";
        std::cin >> s;
        if (s == "0") break;
        cv::Mat img = cv::imread("projekt/" + s + ".jpg"), obj;
        if (img.empty())
        {
            std::cout << "Files not found\n";
            return 0;
        }
        cv::resize(img, obj, cv::Size(1280, 720));
        LicensePlateFinder finder(obj);
        std::cout << "Searching for license plates\n";
        std::cout << "Warning: this step may take up to 5 minutes\n";
        finder.work();
        //finder.boxes.push_back(cv::Rect(cv::Point(267, 360), cv::Point(267 + 597, 360 + 147)));
        for each (cv::Rect box in finder.boxes)
        {
            std::cout << "Reading license plate\n";
            work(s, box);
        }
    }
    return 0;
}

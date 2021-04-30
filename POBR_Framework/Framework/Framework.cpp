#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include "Zad2.h"
#define at at<cv::Vec3b>

void zad1(cv::Mat& img)
{
    int S=0, L=0;
    Tools::getSL(S, L, img);
    std::cout << "S=" << S << ", L=" << L << ", W3=" << Tools::W3(S, L) << ", M1=" << Tools::M1(img, S) << ", M7=" << Tools::M7(img, S) << std::endl;
}

int main(int, char *[]) {
    std::string names[] = { "elipsa.dib", "elipsa1.dib", "kolo.dib", "prost.dib", "troj.dib" }; //prost ob 392
    std::cout << "Start ..." << std::endl;
    std::cout << "Zadanie 1:" << std::endl;
    for each (auto file in names)
    {
        std::cout << "Plik " << file << "  ";
        cv::Mat image = cv::imread(file);
        zad1(image);
        //cv::imshow("", image);
        //cv::waitKey();
        //cv::destroyAllWindows();
    }
    std::string names_2[] = { "strzalki_1.dib", "strzalki_2.dib" };
    std::cout << "Zadanie 2:" << std::endl;
    for each (auto file in names_2)
    {
        std::cout << "Plik " << file << std::endl;
        Zad2 z(cv::imread(file));
    }
    
    return 0;
}

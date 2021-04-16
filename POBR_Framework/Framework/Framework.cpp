#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include "MaskFilter.h"
#include "RankFilter.h"

int main(int, char *[]) {
    std::cout << "Start ..." << std::endl;
    std::cout << "Zadanie 1:" << std::endl;
    cv::Mat image = cv::imread("Lena.bmp");
    //quarterImg(image);
    //float mask[5][5] = { {1,1,1,1,1}, {1,1,1,1,1}, {1,1,1,1,1}, {1,1,1,1,1}, {1,1,1,1,1} };
    float mask[5][5] = { {-1,-1,-1,-1,-1}, {-1,-1,-1,-1,-1}, {-1,-1,24,-1,-1}, {-1,-1,-1,-1,-1}, {-1,-1,-1,-1,-1} };
    MaskFilter f(image, mask);
    cv::Mat filtered = f.filter();
    cv::imshow("filtered", filtered);
    cv::imshow("original", image);
    cv::imwrite("lena_convfiltered.bmp", filtered);
    filtered.release();
    std::cout << "Aby kontynuowaæ, naciœnij dowolny klawisz na oknie z OpenCV" << std::endl;
    cv::waitKey();
    cv::destroyAllWindows();
    std::cout << "Zadanie 2:" << std::endl;
    std::cout << "Podaj n (rozmiar okna rankingowego: 2n+1 x 2n+1): ";
    int n, idx, size;
    std::cin >> n;
    std::cout << "Podaj indeks filtru: ";
    std::cin >> idx;
    size = (2 * n + 1) * (2 * n + 1);
    while (idx >= size)
    {
        std::cout << "B£¥D: indeks nie mieœci siê w oknie rankingowym. Podaj liczbê od 0 do " << size-1 << std::endl;
        std::cout << "Podaj indeks filtru: ";
        std::cin >> idx;
    }
    RankFilter rf(image, n, idx);
    filtered = rf.filter();
    cv::imshow("filtered", filtered);
    cv::imshow("original", image);
    cv::imwrite("lena_rankfiltered.bmp", filtered);
    filtered.release();
    cv::waitKey();
    cv::destroyAllWindows();
    std::cout << "Obrazy wynikowe zosta³y zapisane jako lena_convfiltered oraz lena_rankfiltered" << std::endl;
    return 0;
}

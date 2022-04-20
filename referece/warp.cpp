#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int, char**) {
    Mat image = imread("../../cards.png");

    namedWindow("output", WINDOW_AUTOSIZE);
    moveWindow("output", 400, 200);

    Point2f src[4] = {{0,0},{100,0},{0,100},{200,100}};
    Point2f dst[4] = {{0,0},{100,0},{0,100},{100,100}};

    Mat matric = getPerspectiveTransform(src, dst);
    warpPerspective(image, image, matric, Size(100, 100));

    imshow("output", image);

    waitKey(0);

    return 0;
}

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int, char**) {
    Mat image = imread("../../cards.png");
    VideoCapture cap(1);

    namedWindow("output", WINDOW_AUTOSIZE);
    moveWindow("output", 400, 200);

    while (true){
        cap.read(image);
        cvtColor(image, image, COLOR_BGR2HSV);

        imshow("output", image);
        if (waitKey(50) >= 0) {
            break;
        }
    }
    return 0;
}

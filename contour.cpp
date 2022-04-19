#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void getContrours(Mat img){

    Mat gray, canny;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, canny, 50, 100);

    imshow("canny", canny);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(canny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    drawContours(img, contours, -1, Scalar(0, 0, 255), 5);

    vector<vector<Point>> conPoly(contours.size());

    for(int i = 0; i < contours.size(); i++){
        approxPolyDP(contours[i], conPoly[i], 10, true);
    }

    drawContours(img, conPoly, -1, Scalar(0, 255, 0), 1);

}

int main(int, char**) {
    Mat image = imread("../../shapes.png");

    namedWindow("output", WINDOW_AUTOSIZE);
    moveWindow("output", 400, 200);

    getContrours(image);
    imshow("output", image);

    waitKey(0);
    return 0;
}

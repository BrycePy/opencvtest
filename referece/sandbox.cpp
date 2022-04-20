#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int, char**) {
    Mat image;
    VideoCapture cap(1);

    Mat blank(512, 512, CV_8UC3, Scalar(0, 0, 0));

    namedWindow("Webcam", WINDOW_AUTOSIZE);
    moveWindow("Webcam", 400, 200);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    while (true) {
        cap.read(image);

        // cvtColor(image, image, COLOR_BGR2GRAY);

        // GaussianBlur(image, image, Size(3, 3), 5, 0);
        // Canny(image, image, 50, 100);

        // dilate(image, image, kernel);
        // erode(image, image, kernel);

        circle(image, Point(image.cols / 2, image.rows / 2), 100, Scalar(255, 255, 255), 5);
        circle(image, Point(image.cols / 2, image.rows / 2), 50, Scalar(255, 0, 0), FILLED);

        rectangle(image, Point(image.cols / 2 - 50, image.rows / 2 - 50), Point(image.cols / 2 + 50, image.rows / 2 + 50), Scalar(0, 0, 255), 5);
        line(image, Point(image.cols / 2 - 50, image.rows / 2 - 50), Point(image.cols / 2 + 50, image.rows / 2 + 50), Scalar(0, 255, 0), 5);
        putText(image, "Hello World", Point(100, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

        // resize(image, image, Size(), 0.5, 0.5);

        // Rect crop(100,150,200,200);
        // image = image(crop);

        imshow("Webcam", image);

        if (waitKey(50) >= 0) {
            break;
        }
    }

    return 0;
}

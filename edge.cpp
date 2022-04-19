#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
Mat avgImage(Size(400, 400), CV_8UC3, Scalar(0));

int findTL(vector<Point> p) {
    int min = 0;
    for (int i = 0; i < p.size(); i++) {
        if (p[i].x + p[i].y < p[min].x + p[min].y) {
            min = i;
        }
    }
    return min;
}

Mat getEdge(Mat image, int blurSize, int thresholdValue) {
    Mat imageValue, imageValueNoBlur, imageEdge;
    cvtColor(image, imageValue, COLOR_BGR2HSV);
    extractChannel(imageValue, imageValue, 2);
    imageValueNoBlur = imageValue.clone();
    GaussianBlur(imageValue, imageValue, Size(blurSize, blurSize), 5, 0);
    absdiff(imageValueNoBlur, imageValue, imageEdge);
    threshold(imageEdge, imageEdge, thresholdValue, 255, THRESH_BINARY);
    return imageEdge;
}

Mat getPerspectiveMetrix(Mat image) {
    // Mat imageEdge = getEdge(image, 5, 5);
    Mat imageEdge;
    cvtColor(image, imageEdge, COLOR_BGR2GRAY);
    Canny(imageEdge, imageEdge, 25, 200, 3);
    imshow("imageEdge", imageEdge);
    vector<vector<Point>> contours, scratch;
    vector<Vec4i> hierarchy;
    findContours(imageEdge, contours, hierarchy, RETR_EXTERNAL,
                 CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    for (int i = 0; i < contours.size(); i++)
        approxPolyDP(contours[i], conPoly[i], 30, true);

    float max_area = 10000;
    int max_rect = 0;

    for (int i = 0; i < contours.size(); i++) {
        if (conPoly[i].size() == 4) {
            float area = contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_rect = i;
            }
        }
    }

    Mat imageCont = image.clone();
    Mat imageWarp, imageWarpRGB;

    int size = 400;
    if (max_area > 10000) {
        vector<Point> p = conPoly[max_rect];
        int tl = findTL(p);
        Point2f src[4] = {p[(tl + 4) % 4], p[(tl + 3) % 4], p[(tl + 1) % 4],
                          p[(tl + 2) % 4]};
        Point2f dst[4] = {{0.0, 0.0},
                          {(float)size, 0.0},
                          {0.0, (float)size},
                          {(float)size, (float)size}};

        Mat matric = getPerspectiveTransform(src, dst);
        drawContours(imageCont, conPoly, max_rect, Scalar(0, 255, 0), 3);
        for (int i = 0; i < p.size(); i++) {
            circle(imageCont, p[i], 3, Scalar(0, 100, 255), -1);
            putText(imageCont, to_string(i), p[i], FONT_HERSHEY_SIMPLEX, 1,
                    Scalar(0, 0, 255), 2);
        }

        warpPerspective(image, imageWarpRGB, matric, Size(size, size));
        imshow("imageWarp", imageWarpRGB);

        Rect crop(20, 20, size-40, size-40);
        imageWarpRGB = imageWarpRGB(crop);
        resize(imageWarpRGB, imageWarpRGB, Size(size, size));

        vector<Mat> channels;
        Mat imageFinal;

        Mat imageScratch = getEdge(imageWarpRGB, 5, 3);
        cvtColor(imageWarpRGB, imageWarp, COLOR_BGR2HSV);
        Mat imageOil;

        imageScratch *= 3;
        GaussianBlur(imageScratch, imageScratch, Size(15, 15), 0, 0);
        threshold(imageScratch, imageScratch, 20, 255, THRESH_BINARY);

        findContours(imageScratch, scratch, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

        extractChannel(imageWarp, imageOil, 1);
        imageOil *= 3;
        GaussianBlur(imageOil, imageOil, Size(5, 5), 0, 0);
        threshold(imageOil, imageOil, 70, 255, THRESH_BINARY);

        Mat imageS = Mat::zeros(Size(imageWarp.rows, imageWarp.cols), CV_8UC1);
        ;

        channels.push_back(imageS);
        channels.push_back(imageOil);
        channels.push_back(imageScratch);

        merge(channels, imageFinal);

        avgImage = avgImage * 0.9 + imageFinal * 0.2;

        for (int i = 0; i < scratch.size(); i++) {
            float area = contourArea(scratch[i]);
            if (area > 200) {
                drawContours(imageWarpRGB, scratch, i, Scalar(255, 255, 0), 2);
                Rect bbox = boundingRect(scratch[i]);
                putText(imageWarpRGB, "SAND HERE", Point(bbox.x, bbox.y),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            }
        }

        imshow("imageFinal", avgImage * 0.8 + imageWarpRGB);

        imshow("imageCont", imageCont);
        return matric;
    }

    imshow("imageCont", imageCont);
    return Mat();
}

void camDebug(Mat imageBGR) {
    Mat HChannel, SChannel, VChannel, imageHSV;
    cvtColor(imageBGR, imageHSV, COLOR_BGR2HSV);
    resize(imageBGR, imageBGR, Size(), 0.5, 0.5);
    resize(imageHSV, imageHSV, Size(), 0.5, 0.5);
    extractChannel(imageHSV, HChannel, 0);
    extractChannel(imageHSV, SChannel, 1);
    extractChannel(imageHSV, VChannel, 2);
    imshow("hue", HChannel);
    imshow("saturation", SChannel);
    imshow("value", VChannel);
    imshow("rgb", imageBGR);
    imshow("hsv", imageHSV);
}

int main(int, char**) {
    Mat camRGB, camHSV, image, reference, diffImage, diffImageNoBlur, blur;

    VideoCapture cap(1,CAP_MSMF);
    cap.read(camRGB);
    while (true) {
        int64 start = cv::getTickCount();
        cap.read(camRGB);
        // cvtColor(camRGB, camHSV, COLOR_GRAY2BGR);
        
        imshow("camRGB", camRGB);
        // getPerspectiveMetrix(camRGB);
        // camDebug(camRGB);

        if (waitKey(1) >= 0) {
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;
    }
    return 0;
}

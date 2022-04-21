#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int findTopLeft(vector<Point> p) {
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
    if (image.channels() == 3) {
        cvtColor(image, imageValue, COLOR_BGR2HSV);
        extractChannel(imageValue, imageValue, 2);
    } else {
        imageValue = image.clone();
    }
    imageValueNoBlur = imageValue.clone();
    GaussianBlur(imageValue, imageValue, Size(blurSize, blurSize), 5, 0);
    absdiff(imageValueNoBlur, imageValue, imageEdge);
    threshold(imageEdge, imageEdge, thresholdValue, 255, THRESH_BINARY);
    return imageEdge;
}

Mat edgeDilateKernal = getStructuringElement(MORPH_RECT, Size(3, 3));
Mat oilDilateKernal = getStructuringElement(MORPH_RECT, Size(3, 3));

struct retVals {
    Mat warp, invWarp, imageWarp;
    bool empty() {
        return warp.empty() || invWarp.empty() || imageWarp.empty();
    }
};

retVals getWarp(Mat image, int size, int rotationOffset) {
    // Mat imageEdge;
    // cvtColor(image, imageEdge, COLOR_BGR2GRAY);
    Mat imageEdge, blur;
    GaussianBlur(image, blur, Size(3, 3), 5, 0);
    Canny(blur, imageEdge, 25, 200, 3);
    dilate(imageEdge, imageEdge, edgeDilateKernal);
    imshow("imageEdge" + to_string(rotationOffset), imageEdge);

    // ===== find contours =====
    vector<vector<Point>> contours, scratch;
    vector<Vec4i> hierarchy;
    findContours(imageEdge, contours, hierarchy, RETR_EXTERNAL,
                 CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    for (int i = 0; i < contours.size(); i++)
        approxPolyDP(contours[i], conPoly[i], 30, true);

    // ===== find largest rect contour =====
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
    Mat imageWarpHSV, imageWarpRGB, imageWarpGray;

    if (max_area > 10000) {
        vector<Point> p = conPoly[max_rect];
        int tl = findTopLeft(p) + rotationOffset;
        Point2f src[4] = {p[(tl + 4) % 4], p[(tl + 3) % 4], p[(tl + 1) % 4],
                          p[(tl + 2) % 4]};
        Point2f dst[4] = {{0.0, 0.0},
                          {(float)size, 0.0},
                          {0.0, (float)size},
                          {(float)size, (float)size}};

        Mat warpMatric = getPerspectiveTransform(src, dst);
        Mat invWarpMatric = getPerspectiveTransform(dst, src);
        Mat imageWarp;
        warpPerspective(image, imageWarp, warpMatric, Size(size, size));
        return retVals{warpMatric, invWarpMatric, imageWarp};
    }
    return retVals{Mat(), Mat(), Mat()};
}

struct defectFeatures {
    Mat scratch, oil;
    Mat get3C() {
        vector<Mat> channels;
        Mat imageFinal;
        Mat imageBlank = Mat::zeros(Size(scratch.rows, scratch.cols), CV_8UC1);
        channels.push_back(oil);
        channels.push_back(imageBlank);
        channels.push_back(scratch + oil * 0.5);
        merge(channels, imageFinal);
        return imageFinal;
    }
};

defectFeatures getFeatures(Mat image) {
    Mat imageWarpHSV, imageWarpGray;
    cvtColor(image, imageWarpHSV, COLOR_BGR2HSV);
    cvtColor(image, imageWarpGray, COLOR_BGR2GRAY);

    vector<vector<Point>> scratches;
    vector<Vec4i> hierarchy;

    Mat imageScratch = getEdge(image, 5, 3);
    imageScratch *= 3;
    GaussianBlur(imageScratch, imageScratch, Size(31, 31), 0, 0);
    threshold(imageScratch, imageScratch, 10, 255, THRESH_BINARY);
    findContours(imageScratch, scratches, hierarchy, RETR_TREE,
                 CHAIN_APPROX_SIMPLE);
    // imshow("imageScratch", imageScratch);

    Mat imageOil;
    GaussianBlur(imageWarpGray, imageWarpGray, Size(15, 15), 0, 0);
    imageOil = getEdge(imageWarpGray, 15, 5);
    GaussianBlur(imageOil, imageOil, Size(31, 31), 0, 0);
    threshold(imageOil, imageOil, 35, 255, THRESH_BINARY);
    dilate(imageOil, imageOil, oilDilateKernal);
    // imshow("imageOil", imageOil);
    return defectFeatures{imageScratch, imageOil};
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
    Mat camRGB1, camHSV1;

    VideoCapture cap1(2);
    cap1.read(camRGB1);
    int warpSize = 500;
    Mat imageAverage(Size(warpSize, warpSize), CV_8UC3);

    while (true) {
        int64 start = cv::getTickCount();

        cap1.read(camRGB1);

        retVals warpResult = getWarp(camRGB1, warpSize, 0);

        if (!warpResult.empty()) {
            imshow("imageWarp", warpResult.imageWarp);
        } else {
            imshow("imageWarp", camRGB1);
        }

        if (!warpResult.empty()) {
            defectFeatures result = getFeatures(warpResult.imageWarp);
            ;
            Mat feature = result.get3C();
            imshow("features", feature);

            imageAverage = imageAverage * 0.9 + feature * 0.1;
            imshow("featuresComb", imageAverage);
        }

        if (waitKey(1) >= 0) {
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;
    }
    return 0;
}

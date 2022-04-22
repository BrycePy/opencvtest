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
    if(image.channels() == 3) {
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

struct warpVals {
    Mat warp, invWarp, imageWarp;
};

warpVals getWarp(Mat image, int size, int rotationOffset){
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
        return warpVals {warpMatric, invWarpMatric, imageWarp};
    }
    return warpVals {Mat(), Mat(), Mat()};
}

struct defectFeatures{
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

defectFeatures getFeatures(Mat image){
    Mat imageWarpHSV, imageWarpGray;
    cvtColor(image, imageWarpHSV, COLOR_BGR2HSV);
    cvtColor(image, imageWarpGray, COLOR_BGR2GRAY);

    vector<vector<Point>> scratches;
    vector<Vec4i> hierarchy;

    Mat imageScratch = getEdge(image, 5, 3);
    imageScratch *= 3;
    GaussianBlur(imageScratch, imageScratch, Size(31, 31), 0, 0);
    threshold(imageScratch, imageScratch, 10, 255, THRESH_BINARY);
    findContours(imageScratch, scratches, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
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
    Mat camRGB2, camHSV2;

    VideoCapture cap1(2);
    cap1.read(camRGB1);

    VideoCapture cap2(3);
    cap2.read(camRGB2);

    int warpSize = 500;

    Mat imageAverage(Size(warpSize, warpSize), CV_8UC3);

    while (true) {
        int64 start = cv::getTickCount();

        cap1.read(camRGB1);
        cap2.read(camRGB2);

        // imshow("cam1", camRGB1);
        // imshow("cam2", camRGB2);

        warpVals result;
        result = getWarp(camRGB1, warpSize, 1);
        Mat warpMatric1 = result.warp;
        Mat invWarpMatric1 = result.invWarp;
        Mat imageWarp1 = result.imageWarp;
        
        result = getWarp(camRGB2, warpSize, 0);
        Mat warpMatric2 = result.warp;
        Mat invWarpMatric2 = result.invWarp;
        Mat imageWarp2 = result.imageWarp;

        if(!imageWarp1.empty()){
            imshow("imageWarp1", imageWarp1);
        }else{
            imshow("imageWarp1", camRGB1);
        }

        if(!imageWarp2.empty()){
            imshow("imageWarp2", imageWarp2);
        }else{
            imshow("imageWarp2", camRGB2);
        }
    
        if(!warpMatric1.empty() && !warpMatric2.empty()){
            defectFeatures result;
            Mat feature1, feature2;
    
            result = getFeatures(imageWarp1);
            feature1 = result.get3C();
            imshow("features1",feature1);

            result = getFeatures(imageWarp2);
            feature2 = result.get3C();
            imshow("features2",feature2);

            imageAverage = imageAverage * 0.9 + (feature1 + feature2) * 0.5 * 0.1;
            imshow("featuresComb",imageAverage);
        }


        if (waitKey(1) >= 0) {
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;
    }
    return 0;
}

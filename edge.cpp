#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));
Mat kernelLarge = getStructuringElement(MORPH_ELLIPSE, Size(51, 51));
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

Mat processFeed(Mat image) {
    // ===== generate edge image =====
    // Mat imageEdge = getEdge(image, 5, 5);
    Mat imageEdge;
    cvtColor(image, imageEdge, COLOR_BGR2GRAY);
    Canny(imageEdge, imageEdge, 25, 200, 3);
    // imshow("imageEdge", imageEdge);

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

        Mat warpMatric = getPerspectiveTransform(src, dst);
        Mat invWarpMatric = getPerspectiveTransform(dst, src);
        drawContours(imageCont, conPoly, max_rect, Scalar(0, 255, 0), 3);
        for (int i = 0; i < p.size(); i++) {
            circle(imageCont, p[i], 3, Scalar(0, 100, 255), -1);
            putText(imageCont, to_string(i), p[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }

        warpPerspective(image, imageWarpRGB, warpMatric, Size(size, size));
        imshow("imageWarp", imageWarpRGB);

        Rect crop(5, 5, size - 10, size - 10);
        imageWarpRGB = imageWarpRGB(crop);
        resize(imageWarpRGB, imageWarpRGB, Size(size, size));

        cvtColor(imageWarpRGB, imageWarpHSV, COLOR_BGR2HSV);
        cvtColor(imageWarpRGB, imageWarpGray, COLOR_BGR2GRAY);

        Mat imageScratch = getEdge(imageWarpRGB, 5, 3);
        imageScratch *= 3;
        GaussianBlur(imageScratch, imageScratch, Size(31, 31), 0, 0);
        threshold(imageScratch, imageScratch, 10, 255, THRESH_BINARY);
        findContours(imageScratch, scratch, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        imshow("imageScratch", imageScratch);

        Mat imageOil;
        GaussianBlur(imageWarpGray, imageWarpGray, Size(15, 15), 0, 0);
        imageOil = getEdge(imageWarpGray, 15, 5);
        GaussianBlur(imageOil, imageOil, Size(31, 31), 0, 0);
        threshold(imageOil, imageOil, 35, 255, THRESH_BINARY);
        dilate(imageOil, imageOil, kernelLarge);
        imshow("imageOil", imageOil);

        Mat imageBlank = Mat::zeros(Size(imageWarpHSV.rows, imageWarpHSV.cols), CV_8UC1);

        Mat imageFinal;
        vector<Mat> channels;
        channels.push_back(imageOil);
        channels.push_back(imageBlank);
        channels.push_back(imageOil * 0.5);
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

        Mat imageFinalOverlay = avgImage * 0.8 + imageWarpRGB;
        imshow("imageFinal", imageFinalOverlay);

        warpPerspective(avgImage, imageFinalOverlay, invWarpMatric, Size(image.cols, image.rows));

        imshow("imageInvWarpFinal", image + imageFinalOverlay);
        imshow("imageCont", imageCont);
        return warpMatric;
    }

    imshow("imageInvWarpFinal", image);
    imshow("imageCont", imageCont);
    return Mat();
}

struct warpVals {
    Mat warp, invWarp;
};

warpVals getWarp(Mat image, int size, int rotationOffset){
    Mat imageEdge;
    cvtColor(image, imageEdge, COLOR_BGR2GRAY);
    Canny(imageEdge, imageEdge, 25, 200, 3);
    // imshow("imageEdge", imageEdge);

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
        int tl = findTL(p) + rotationOffset; 
        Point2f src[4] = {p[(tl + 4) % 4], p[(tl + 3) % 4], p[(tl + 1) % 4],
                          p[(tl + 2) % 4]};
        Point2f dst[4] = {{0.0, 0.0},
                          {(float)size, 0.0},
                          {0.0, (float)size},
                          {(float)size, (float)size}};

        Mat warpMatric = getPerspectiveTransform(src, dst);
        Mat invWarpMatric = getPerspectiveTransform(dst, src);
        return warpVals {warpMatric, invWarpMatric};
    }
    return warpVals {Mat(), Mat()};
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
    Mat camRGB2, camHSV2;

    VideoCapture cap(2);
    cap.read(camRGB);

    VideoCapture cap2(3);
    cap2.read(camRGB2);

    Mat imageAverage(Size(500, 500), CV_32FC3);

    while (true) {
        int64 start = cv::getTickCount();

        cap.read(camRGB);
        cap2.read(camRGB2);

        imshow("cam1", camRGB);
        imshow("cam2", camRGB2);

        // processFeed(camRGB);
        warpVals result = getWarp(camRGB, 500, 1);
        Mat warpMatric1 = result.warp;
        Mat invWarpMatric1 = result.invWarp;
        
        Mat warp1, invWarp1;
        Mat warp2, invWarp2;

        if(!warpMatric1.empty()){
            warpPerspective(camRGB, warp1, warpMatric1, Size(500, 500));
            imshow("cam1Warp", warp1);
            warpPerspective(warp1, invWarp1, invWarpMatric1, Size(camRGB.cols, camRGB.rows));
            imshow("cam1InvWarp", invWarp1);
        }


        result = getWarp(camRGB2, 500, 0);
        Mat warpMatric2 = result.warp;
        Mat invWarpMatric2 = result.invWarp;
        
        if(!warpMatric2.empty()){
            warpPerspective(camRGB2, warp2, warpMatric2, Size(500, 500));
            imshow("cam2Warp", warp2);
            warpPerspective(warp2, invWarp2, invWarpMatric2, Size(camRGB2.cols, camRGB2.rows));
            imshow("cam2InvWarp", invWarp2);
        }

        if(!warpMatric1.empty() && !warpMatric2.empty()){
            warp1.convertTo(warp1, CV_32FC3);
            warp2.convertTo(warp2, CV_32FC3);
            imageAverage = imageAverage * 0.9 + warp1 * 0.05 + warp2 * 0.05;

            Mat imageAverageRGB;
            imageAverageRGB = imageAverage.clone();
            imageAverageRGB.convertTo(imageAverageRGB, CV_8UC3);
            imshow("camCombine", imageAverageRGB);

            Mat imageEdge = getEdge(imageAverage, 5, 5);
            imshow("camCombineEdge", imageEdge);

            Mat combHSV;
            cvtColor(imageAverageRGB, combHSV, COLOR_BGR2HSV);
            Mat combH, combS, combV;
            extractChannel(combHSV, combH, 0);
            extractChannel(combHSV, combS, 1);
            extractChannel(combHSV, combV, 2);
            Mat white(Size(combHSV.cols, combHSV.rows), CV_8UC1, Scalar(100));



            Mat imageFinal;
            vector<Mat> channels;
            channels.push_back(combH);
            channels.push_back(white);
            channels.push_back(combV);
            merge(channels, imageFinal);

            cvtColor(imageFinal, imageFinal, COLOR_HSV2BGR);
            imshow("camCombineHSV", imageFinal);


        }


        if (waitKey(10) >= 0) {
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;
    }
    return 0;
}

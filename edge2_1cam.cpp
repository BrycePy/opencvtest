#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat edgeDilateKernal = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat oilDilateKernal = getStructuringElement(MORPH_ELLIPSE, Size(31, 31));
Mat NoiseDilateKernal = getStructuringElement(MORPH_ELLIPSE, Size(51, 51));


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

struct warpVals {
    Mat warp, invWarp, imageWarp;
    int cropAmount = 0;
    bool empty() {
        return warp.empty() || invWarp.empty() || imageWarp.empty();
    }
    void cropIn(int amount) {
        cropAmount += amount;
        Rect crop(amount, amount, imageWarp.cols - amount*2, imageWarp.cols - amount*2);
        imageWarp = imageWarp(crop);
    }
    void reverse() {
        Mat originalSize(imageWarp.rows + cropAmount*2, imageWarp.cols + cropAmount*2, imageWarp.type());
        Mat insetImage(originalSize, Rect(cropAmount, cropAmount, imageWarp.cols, imageWarp.rows));
        originalSize = Scalar(0, 0, 0);
        imageWarp.copyTo(insetImage);
        imageWarp = originalSize.clone();
        cropAmount = 0;
    }
    Mat reverse(Mat image) {
        Mat originalSize(image.rows + cropAmount*2, image.cols + cropAmount*2, image.type());
        Mat insetImage(originalSize, Rect(cropAmount, cropAmount, image.cols, image.rows));
        originalSize = Scalar(0, 0, 0);
        image.copyTo(insetImage);
        return originalSize.clone();
    }
    Mat invert(Mat bg, Mat image){
        Mat inverted;
        warpPerspective(image, inverted, invWarp, Size(bg.cols, bg.rows));
        return inverted;
    }
};

warpVals getWarp(Mat image, int size, int rotationOffset, int camNum) {
    // Mat imageEdge;
    // cvtColor(image, imageEdge, COLOR_BGR2GRAY);
    Mat imageEdge, blur;
    GaussianBlur(image, blur, Size(3, 3), 5, 0);
    Canny(blur, imageEdge, 25, 200, 3);
    dilate(imageEdge, imageEdge, edgeDilateKernal);
    imshow("imageEdge" + to_string(camNum), imageEdge);

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
        return warpVals{warpMatric, invWarpMatric, imageWarp};
    }
    return warpVals{Mat(), Mat(), Mat()};
}

struct defectFeatures {
    Mat scratch, oil, hole;

    Mat getOverlay() {
        vector<Mat> channels;

        Mat scratchTemp;
        dilate(scratch, scratchTemp, oilDilateKernal);
        oil -= scratchTemp;
        dilate(oil, oil, oilDilateKernal);
        dilate(oil, oil, oilDilateKernal);

        Mat imageFinal;
        Mat imageBlank = Mat::zeros(Size(scratch.rows, scratch.cols), CV_8UC1);
        channels.push_back(oil);
        channels.push_back(hole * 0.5);
        channels.push_back(scratch + oil * 0.3 + hole * 0.5);
        merge(channels, imageFinal);
        return imageFinal;
    }

    float getScratchAmount() {
        return (float)countNonZero(scratch) / (scratch.rows * scratch.cols);
    }

    float getOilAmount() {
        return (float)countNonZero(oil) / (oil.rows * oil.cols);
    }

    float getTotalAmount() {
        Mat defect = scratch + oil;
        return (float)countNonZero(defect) / (defect.rows * defect.cols);
    }

};

defectFeatures getFeatures(Mat image) {
    Mat imageWarpHSV, imageWarpGray;
    cvtColor(image, imageWarpHSV, COLOR_BGR2HSV);
    cvtColor(image, imageWarpGray, COLOR_BGR2GRAY);

    vector<vector<Point>> scratches;
    vector<Vec4i> hierarchy;

    Mat imageScratch = getEdge(image, 5, 5);
    imshow("imageScratch", imageScratch);
    imageScratch *= 3;
    GaussianBlur(imageScratch, imageScratch, Size(15, 15), 0, 0);
    threshold(imageScratch, imageScratch, 20, 255, THRESH_BINARY);
    findContours(imageScratch, scratches, hierarchy, RETR_TREE,
                 CHAIN_APPROX_SIMPLE);
    // imshow("imageScratch", imageScratch);

    Mat imageOil;
    GaussianBlur(imageWarpGray, imageWarpGray, Size(25, 25), 0, 0);
    imageOil = getEdge(imageWarpGray, 15, 5);
    GaussianBlur(imageOil, imageOil, Size(31, 31), 0, 0);
    threshold(imageOil, imageOil, 35, 255, THRESH_BINARY);
    dilate(imageOil, imageOil, oilDilateKernal);
    imshow("imageOil", imageOil);

    Mat imageHole(image.rows, image.cols, CV_8UC1, Scalar(0));
    Mat imageV;
    extractChannel(imageWarpHSV, imageV, 2);
    rotate(imageV, imageV, ROTATE_180);
    Rect crop(280, 50, 150, 150);
    Mat holeCrop = imageV(crop);

    double min, max;
    bool p1, p2;

    crop = Rect(40, 40, 20, 20);
    Mat holeCrop1 = holeCrop(crop);
    minMaxIdx(holeCrop1, &min, &max);
    p1 = max - min > 45;
        
    crop = Rect(70, 68, 20, 20);
    Mat holeCrop2 = holeCrop(crop);
    minMaxIdx(holeCrop2, &min, &max);
    p2 = max - min > 45;

    if(!(p1 || p2)) {
        circle(imageHole, Point(280 + 65, 50 + 73), 30, Scalar(255), -1);
    }

    rotate(imageHole, imageHole, ROTATE_180);

    imshow("imageHole", imageHole);

    return defectFeatures{imageScratch, imageOil, imageHole};
}

void camDebug(Mat imageBGR) {
    Mat HChannel, SChannel, VChannel, imageHSV;
    cvtColor(imageBGR, imageHSV, COLOR_BGR2HSV);
    resize(imageBGR, imageBGR, Size(), 0.25, 0.25);
    resize(imageHSV, imageHSV, Size(), 0.25, 0.25);
    extractChannel(imageHSV, HChannel, 0);
    extractChannel(imageHSV, SChannel, 1);
    extractChannel(imageHSV, VChannel, 2);
    imshow("hue", HChannel);
    imshow("saturation", SChannel);
    imshow("value", VChannel);
    imshow("rgb", imageBGR);
    imshow("hsv", imageHSV);
}

Mat normalizeLight(Mat image) {
    Rect crop1(0, 0, image.cols, 5);
    Rect crop2(image.cols - 5, 0, 5, image.rows);
    Rect crop3(0, 0, 5, image.rows);
    Rect crop4(0, image.rows - 5, image.cols, 5);

    Mat imageCrop1 = image(crop1);
    Mat imageCrop2 = image(crop2);
    Mat imageCrop3 = image(crop3);
    Mat imageCrop4 = image(crop4);

    GaussianBlur(imageCrop1, imageCrop1, Size(5, 5), 0, 0);
    GaussianBlur(imageCrop2, imageCrop2, Size(5, 5), 0, 0);
    GaussianBlur(imageCrop3, imageCrop3, Size(5, 5), 0, 0);
    GaussianBlur(imageCrop4, imageCrop4, Size(5, 5), 0, 0);

    resize(imageCrop1, imageCrop1, Size(image.cols, image.rows));
    resize(imageCrop2, imageCrop2, Size(image.cols, image.rows));
    resize(imageCrop3, imageCrop3, Size(image.cols, image.rows));
    resize(imageCrop4, imageCrop4, Size(image.cols, image.rows));

    Mat comb = (imageCrop1/ 4 + imageCrop2/ 4 + imageCrop3/ 4 + imageCrop4/ 4);
    GaussianBlur(comb, comb, Size(51, 51), 0, 0);

    Mat imageTemp;
    image.convertTo(imageTemp, CV_32F);
    comb.convertTo(comb, CV_32F);

    comb *= 2;
    Scalar colorDiff = (mean(imageCrop1) + mean(imageCrop2)+ mean(imageCrop3)+ mean(imageCrop4)) / 4 - mean(comb);
    comb += colorDiff;
    comb.convertTo(comb, CV_8U);
    return comb;
}

int main(int, char**) {
    Mat camRGB1, camHSV1;
    Mat camRGB2, camHSV2;

    VideoCapture cap1(2);

    namedWindow("config", WINDOW_AUTOSIZE);

    int multiplyer = 90;
    createTrackbar("min", "config", &multiplyer, 1000);

    cap1.read(camRGB1);
    int warpSize = 480;
    Mat imageAverage(Size(warpSize, warpSize), CV_8UC3);
    Mat imageAverage2(Size(warpSize, warpSize), CV_8UC1);

    // int taille = warpSize;    
    // Mat image(taille,taille,CV_8UC1);
    // for(int y = 0; y < taille; y++){
    //     Vec3b color((y*255)/taille);
    //     for(int x = 0; x < taille; x++)
    //         image.at<Vec3b>(y,x) = color;
    // }
    Mat gradient = imread("../../gradient.png");
    cvtColor(gradient, gradient, COLOR_BGR2GRAY);

    Mat avgLight(Size(warpSize, warpSize), CV_32FC3);
    // Mat img(warpSize, warpSize, CV_8UC3);
    // RNG rng(12345);

    double defectAmountAverage = 0;

    while (true) {
        int64 start = cv::getTickCount();

        cap1.read(camRGB1);
        warpVals warpResult = getWarp(camRGB1, warpSize, 0, 1);

        {
            // if (!warpResult.empty()) {
            //     imshow("imageWarp", warpResult.imageWarp);

            //     Mat imageWarpRGB = warpResult.imageWarp;
            //     // cvtColor(warpResult.imageWarp, imageWarpHSV, COLOR_BGR2HSV);

            //     Rect crop(5, 5, warpSize - 10, warpSize - 10);
            //     imageWarpRGB = imageWarpRGB(crop);
            //     resize(imageWarpRGB, imageWarpRGB, Size(warpSize, warpSize));

            //     Mat comb = normalizeLight(imageWarpRGB);
            //     Mat cam1Warp = warpResult.imageWarp.clone();

            //     cam1Warp.convertTo(cam1Warp, CV_32FC3);
            //     comb.convertTo(comb, CV_32FC3);
            //     avgLight = avgLight * 0.95 + comb * 0.05;
            //     Mat refDiff = (avgLight - cam1Warp) * ((double)multiplyer / 100) + 128;
            //     refDiff.convertTo(refDiff, CV_8UC3);

            //     imshow("refDiff", refDiff);

            //     Mat refDiffEdge = getEdge(refDiff, 15, 7);
            //     // erode(refDiffEdge, refDiffEdge, edgeDilateKernal);

            //     imageAverage2 = imageAverage2 * 0.9 + refDiffEdge * 0.1;
            //     imshow("refDiffEdge", imageAverage2);

            //     Mat temp;
            //     avgLight.convertTo(temp, CV_8UC3);
            //     // comb.convertTo(comb, CV_8UC3);
            //     imshow("comb", temp);


            // } else {
            //     imshow("imageWarp", camRGB1);
            // }
        }

        if (!warpResult.empty()) {
            warpResult.cropIn(10);
            Mat cam1Warp = warpResult.imageWarp.clone();
            Mat comb = normalizeLight(cam1Warp);
            cam1Warp.convertTo(cam1Warp, CV_32FC3);
            comb.convertTo(comb, CV_32FC3);

            if(avgLight.rows != comb.rows)
                avgLight = comb.clone();

            avgLight = avgLight * 0.95 + comb * 0.05;

            Mat refDiff = (cam1Warp - avgLight) * ((double)multiplyer / 100) + 128;
            refDiff.convertTo(refDiff, CV_8UC3);
            Mat refDiffV;
            cvtColor(refDiff, refDiffV, COLOR_BGR2HSV);
            extractChannel(refDiffV, refDiffV, 2);
            imshow("refDiff", refDiffV);

            defectFeatures result = getFeatures(refDiff);

            Mat feature = result.getOverlay();
            imshow("features", feature);
            feature = warpResult.reverse(feature);

            imageAverage = imageAverage * 0.8 + feature * 0.2;
            imshow("featuresComb", imageAverage);

            float defectAmount = result.getTotalAmount() * 100;
            defectAmountAverage = defectAmountAverage * 0.9 + defectAmount * 0.1;
    
            Mat imageOverlay;
            if(defectAmountAverage > 0.05) {
                imageOverlay = warpResult.invert(camRGB1, imageAverage);
                putText(imageOverlay, "Defect: " + to_string(defectAmountAverage) + "%", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
            }else{
                Mat greenOverlay(imageAverage.rows, imageAverage.cols, CV_8UC3, Scalar(0, 100, 0));
                imageOverlay = warpResult.invert(camRGB1, greenOverlay + imageAverage);
            }

            Mat output = imageOverlay * 0.9 + camRGB1 * 0.8;
            resize(output, output, Size(), 2, 2);
            imshow("imageOverlay", output);

        }else{
            Mat output = camRGB1.clone();
            resize(output, output, Size(), 2, 2);
            imshow("imageOverlay", output);
        }

        if (waitKey(1) >= 0) {
            break;
        }

        double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
        std::cout << "FPS : " << fps << std::endl;
    }
    return 0;
}

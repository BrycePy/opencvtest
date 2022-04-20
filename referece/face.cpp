#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

int main(int, char**) {
    Mat image, cam, face_crop;
    VideoCapture cap(2);

    CascadeClassifier faceC;
    faceC.load("../../haarcascade_frontalface_default.xml");

    namedWindow("output", WINDOW_AUTOSIZE);
    moveWindow("output", 400, 200);
    namedWindow("face", WINDOW_AUTOSIZE);
    moveWindow("face", 1000, 200);

    int factor = 4;

    Rect face(0, 0, 0, 0);
    while (true){
        cap.read(cam);
        resize(cam, image, Size(), 1.0/factor, 1.0/factor);
        
        vector<Rect> faces;
        faceC.detectMultiScale(image, faces);

        if(faces.size()==1){
            face.x = (int)(0.75*faces[0].x*factor + 0.25*face.x);
            face.y = (int)(0.75*faces[0].y*factor + 0.25*face.y);
            face.width = (int)(0.2*faces[0].width*factor + 0.8*face.width);
            face.height = (int)(0.2*faces[0].height*factor + 0.8*face.height);

            rectangle(cam, face, Scalar(0, 0, 255), 2);
            
            face_crop = cam(face);
            resize(face_crop, face_crop, Size(300,300));
            imshow("face", face_crop);
        }

        // cout << __cplusplus << endl;

        imshow("output", cam);
        if (waitKey(30) >= 0) {
            break;
        }
    }
    return 0;
}

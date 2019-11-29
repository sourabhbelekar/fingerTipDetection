#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const Scalar WHITE = Scalar(255,255,255);
const Scalar BLACK = Scalar(0,0,0);

void main()
{

VideoCapture cameraStream(0);
Mat originalFrame,originalGray;
CascadeClassifier face_cascade;

if(!cameraStream.isOpened()){
    cout<< "Cannot open Camera";
    exit(-1);
}

if(!face_cascade.load("haarcascade_frontalface_alt.xml"))
{
    cout << "Unable to load the face detector!!!!" << endl;
    exit (-1);
}

while(1){
    cameraStream >> originalFrame;

    cvtColor(originalFrame, originalGray, cv::COLOR_BGR2GRAY);


    vector<Rect> faces;
    face_cascade.detectMultiScale( originalGray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    if(faces.size() > 0)
        for(int i=0; i<faces.size(); i++)
            rectangle(originalFrame, faces[i], BLACK,-1);




    imshow("Camera Stream",originalFrame);
    if(waitKey(30) >= 0) break;
}

}

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void main()
{

VideoCapture cameraStream(0);
Mat originalFrame;

if(!cameraStream.isOpened()){
    cout<< "Cannot open Camera";
    exit(-1);
}

while(1){
    cameraStream >> originalFrame;

    imshow("Camera Stream",originalFrame);
    if(waitKey(30) >= 0) break;
}

}

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const Scalar WHITE = Scalar(255,255,255);
const Scalar BLACK = Scalar(0,0,0);

int isSkinPixel(Vec3d);

void main()
{

VideoCapture cameraStream(0);
Mat originalFrame,originalGray,noFacesFrame;
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

cameraStream >> originalFrame;

Mat noFaceDoublePrecision(Size(originalFrame.cols, originalFrame.rows), CV_64FC3);
Mat imgSegmented(Size(originalFrame.cols, originalFrame.rows), CV_8UC1, BLACK);
Mat imgSegmentedSmooth(Size(originalFrame.cols, originalFrame.rows), CV_8UC1, BLACK);



while(1){
    cameraStream >> originalFrame;

    cvtColor(originalFrame, originalGray, cv::COLOR_BGR2GRAY);


    vector<Rect> faces;
    face_cascade.detectMultiScale( originalGray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    noFacesFrame = originalFrame.clone();
    if(faces.size() > 0)
        for(int i=0; i<faces.size(); i++)
            rectangle(noFacesFrame, faces[i], BLACK,-1);

    noFacesFrame.convertTo(noFaceDoublePrecision,CV_64FC3,1/255.0);

    for(int i=0;i<noFaceDoublePrecision.total();i++)
        imgSegmented.at<uchar>(i) = isSkinPixel(noFaceDoublePrecision.at<Vec3d>(i));


    medianBlur(imgSegmented,imgSegmentedSmooth,5);

    bitwise_and(originalFrame,imgSegmentedSmooth,displayImg);

    imshow("Camera Stream",imgSegmentedSmooth);
    if(waitKey(30) >= 0) break;
}

}

//based on paper published at https://www.sciencedirect.com/science/article/pii/S0165168409001686
int isSkinPixel(Vec3d pixel){

    double customGray,NoRed,error;
    customGray = pixel.val[0]*0.140209042551032500 + pixel.val[1]*0.587043074451121360 + pixel.val[2]*0.298936021293775390;
    NoRed = max(pixel.val[0],pixel.val[1]);

    error = customGray-NoRed;

 if(error<=0.1177 && error>=0.02511)
    return 255;

 return 0;
}

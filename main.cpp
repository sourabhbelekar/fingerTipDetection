#include <iostream>
#include <opencv2/opencv.hpp>

#define PICTURE_MODE 1
#define VIDEO_MODE 2
using namespace std;
using namespace cv;

const Scalar WHITE = Scalar(255,255,255);
const Scalar BLACK = Scalar(0,0,0);

int isSkinPixel(Vec3d);
vector<vector<Point>> removeNoiseContours(vector<vector<Point>>,int);
void main()
{

int mode = VIDEO_MODE;

VideoCapture cameraStream(0);//("vid1.mp4");// = VideoCapture('vid1.mp4');
Mat originalFrame,originalGray,noFacesFrame;
CascadeClassifier face_cascade;
vector<vector<Point> > contours;
vector<Rect> roi;
int maxContours,maxContLine;
int noiseContourArea=1000;


if(mode == VIDEO_MODE && !cameraStream.isOpened()){
    cout<< "Cannot open Camera";
    exit(-1);
}

if(!face_cascade.load("haarcascade_frontalface_alt.xml"))
{
    cout << "Unable to load the face detector!!!!" << endl;
    exit (-1);
}

if(mode==PICTURE_MODE)
    originalFrame = imread("hand5.jpg");
else
    cameraStream >> originalFrame;

Mat noFaceDoublePrecision(Size(originalFrame.cols, originalFrame.rows), CV_64FC3);
Mat imgSegmented(Size(originalFrame.cols, originalFrame.rows), CV_8UC1, BLACK);
Mat imgSegmentedSmooth(Size(originalFrame.cols, originalFrame.rows), CV_8UC1, BLACK);
Mat imgSegmentedSmoothCopy(Size(originalFrame.cols, originalFrame.rows), CV_8UC1, BLACK);



while(1){
    if(mode==VIDEO_MODE)
        cameraStream >> originalFrame;


    cvtColor(originalFrame, originalGray, cv::COLOR_BGR2GRAY);


    vector<Rect> faces;
    face_cascade.detectMultiScale( originalGray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    noFacesFrame = originalFrame.clone();
    if(faces.size() > 0)
        for(int i=0; i<faces.size(); i++)
            rectangle(noFacesFrame, faces[i], BLACK,-1);


    //noFacesFrame = originalFrame.clone();
    noFacesFrame.convertTo(noFaceDoublePrecision,CV_64FC3,1/255.0);

    for(int i=0;i<noFaceDoublePrecision.total();i++)
        imgSegmented.at<uchar>(i) = isSkinPixel(noFaceDoublePrecision.at<Vec3d>(i));

    imwrite("output/segmented.jpg",imgSegmented);
    medianBlur(imgSegmented,imgSegmentedSmooth,3);
//    imgSegmentedSmooth = imgSegmented.clone();
    imgSegmentedSmoothCopy = imgSegmentedSmooth.clone();


    findContours( imgSegmentedSmoothCopy , contours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    contours = removeNoiseContours(contours,noiseContourArea);
    roi.clear();
    for(int i=0;i<contours.size();i++)
        roi.insert(roi.begin()+i,boundingRect(contours[i]));




    for(int i=0;i<roi.size();i++){

        Mat temp,tempCopy,temp1;
        RotatedRect finger;

        temp = imgSegmentedSmooth(roi[i]);

        char filename[20];

        sprintf(filename,"output/aoi_%d.jpg",i);
        imwrite(filename,temp);

        temp1= originalFrame(roi[i]);
        int maxCont=0,maxContLine;
        vector<vector<Point> > tempContours,tempContours1;

        tempContours1.clear();
        tempCopy = temp.clone();

        for(int i=temp.rows-1;i>=0;i=i-2){

            line(tempCopy,Point(0,i),Point(temp.cols,i),BLACK,1);
            line(tempCopy,Point(0,i-1),Point(temp.cols,i-1),BLACK,1);
            findContours(tempCopy , tempContours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
            tempContours1 = removeNoiseContours(tempContours,noiseContourArea);
            if(tempContours1.size()>maxCont){
                maxContLine=i;
                maxCont = tempContours1.size();
                contours.clear();
                contours = tempContours1;
            }
        }
/*
        rectangle(temp,Point(0,maxContLine-5),Point(temp.cols,temp.rows),BLACK,-1);

        char filename_temp[20];

        sprintf(filename_temp,"output/fingers_%d.jpg",i);
        imwrite(filename_temp,temp);

        findContours(temp.clone() , tempContours, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        tempContours1 = removeNoiseContours(tempContours,noiseContourArea);
*/
        tempContours1.clear();
        tempContours1=contours;
        for(int i=0;i<tempContours1.size();i++)
        {
        //    drawContours(temp1,tempContours1,i,WHITE,2);
            finger= minAreaRect(tempContours1[i]);
            Point2f rect_points[4];
            finger.points( rect_points );

            for( int j = 0; j < 4; j++ )
               line( temp1, rect_points[j], rect_points[(j+1)%4], CV_RGB(255,0,0), 2, 8 );

            //sprintf(filename_temp,"output/fingers_marked_%d.jpg",i);

            //imwrite(filename_temp,originalFrame);

            if(finger.size.height>finger.size.width){
                finger.center = (rect_points[1]+rect_points[2])/2 + (rect_points[0]-rect_points[1])/5;
                finger.size.height = (float)(0.4) * finger.size.height;
            }else{

                finger.center = (rect_points[2]+rect_points[3])/2 + (rect_points[0]-rect_points[3])/6;
                finger.size.width = (float)(0.33) * finger.size.width;
            }

            finger.points( rect_points );
                   for( int j = 0; j < 4; j++ )
                      line( temp1, rect_points[j], rect_points[(j+1)%4], CV_RGB(0,0,255), 2, 8 );

            //cout<<"1 X:"<<rect_points[1].x<<" Y:"<< rect_points[1].y<<endl;
            //cout<<"2 X:"<<rect_points[2].x<<" Y:"<< rect_points[2].y<<endl;
            //cout<<"3 X:"<<rect_points[3].x<<" Y:"<< rect_points[3].y<<endl;


        }


    }

    imwrite("output/final_op.jpg",originalFrame);

    imshow("Camera Stream",originalFrame);
    imshow("Segmented",imgSegmentedSmooth);

    if(mode==PICTURE_MODE)
        break;

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

vector<vector<Point>> removeNoiseContours(vector<vector<Point>> contours, int noiseContourArea){
if(contours.size()>1)
    for(int i=contours.size()-1;i>=0;i--)
       if(contourArea(contours[i],false)<noiseContourArea)
           contours.erase(contours.begin()+i);

return contours;

}

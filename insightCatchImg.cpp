//
// Created by Bosen on 28/11/2019.
// 利用insight图传采集由QGC传出标志位的图像
//

#include "insightCatchImg.h"

void imageCatch_insight(vector<Mat> & imageCatchesVector, string rtspAddress, string qgcTxtFolder, string imgFileFolder, int imgTotalNum) {
    //图像输入
    VideoCapture rtspVideoCapture;

    //打开预先定义的rtsp数据流
    rtspVideoCapture.open(rtspAddress);

    //检测rtsp视频数据流是否被正确打开
    while (!rtspVideoCapture.isOpened()) {
        cout << "!!!!! Rtsp Video Capture FAILED!" << endl;
    }

    cout << ">>>> RTSP Video Capture success" << endl;

    Mat imgCaptureMat;  //当前的capture mat
    string thisTxtName;  //当前读入的QGCtxt的文件名称
    int thisImgNum = 0; //当前采集的图像数量
    struct stat buffer; //文件stat buffer

    thisTxtName = qgcTxtFolder + to_string(thisImgNum) + ".txt";

    /**
     * 依据QGC中的txt结果循环采集图像
     */
    for (thisImgNum = 0; thisImgNum < imgTotalNum;) {
        cout << ">>>> Try find txt:" << thisTxtName << endl;

        while (1) {
            //如果查询文件夹qgcTxtFloder发现txt文件
            if (stat(thisTxtName.c_str(), &buffer) == 0) {
                //从视频流中读入图像
                rtspVideoCapture >> imgCaptureMat;

                //imshow("imgCatch", imgCatchMat);
                Mat imgCaptureRemapMat;
                imageReMap(imgCaptureMat, imgCaptureRemapMat);

                //将照片旋转成为竖幅便于拼接
                transpose(imgCaptureRemapMat, imgCaptureRemapMat);
                flip(imgCaptureRemapMat, imgCaptureRemapMat, 1);

                //将该帧图像加入到image_catches Vector中
                imageCatchesVector.push_back(imgCaptureRemapMat);


                //将采集到的图像保存到本机
                string imgFileName = imgFileFolder + to_string(thisImgNum) + "_" + getTimeString_ICI() + ".jpg";
                imwrite(imgFileName, imgCaptureMat);
                cout << ">>>> Catch Image: " << thisImgNum << "success" << endl;

                break;
            }
        }
        thisImgNum = thisImgNum + 1;
        thisTxtName = qgcTxtFolder + to_string(thisImgNum) + ".txt";
    }

}

/***
 * 进行图像畸变矫正
 */
void imageReMap(Mat srcMat, Mat distMat) {
    //图像size
    Size imgSize;
    imgSize = srcMat.size();

    // 设置相机畸变参数
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 1.7045143537356953e+03;
    cameraMatrix.at<double>(0, 1) = 0;
    cameraMatrix.at<double>(0, 2) = 9.5950000000000000e+02;
    cameraMatrix.at<double>(1, 1) = 1.7045143537356953e+03;
    cameraMatrix.at<double>(1, 2) = 5.3950000000000000e+02;

    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -3.1946395398598881e-01;
    distCoeffs.at<double>(1, 0) = 1.9135700460721553e-01;
    distCoeffs.at<double>(2, 0) = 0;
    distCoeffs.at<double>(3, 0) = 0;
    distCoeffs.at<double>(4, 0) = -6.3363712039867705e-03;

    //畸变map计算
    Mat imgMap1, imgMap2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 1, imgSize, 0),
                            imgSize, CV_16SC2, imgMap1, imgMap2);

    Mat imgCalMat(INSIGHT_IN_HEI, INSIGHT_IN_WID, srcMat.type()); //畸变校正后的图像
    Mat imgCalRezMat(INSIGHT_CAL_HEI, INSIGHT_CAL_WID, srcMat.type()); //畸变校正之后黑边去除的图像

    //畸变校正
    remap(srcMat, imgCalMat, imgMap1, imgMap2, INTER_LINEAR);
    //去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
    resizeImage(imgCalMat,imgCalRezMat,INSIGHT_CAL_REZ_TOP,INSIGHT_CAL_REZ_LEFT,INSIGHT_CAL_REZ_HEI,INSIGHT_CAL_REZ_WID);

    distMat = imgCalRezMat.clone();
    cout << ">>>> remap this image success" << endl;
}


/**
 * 返回当前时间(精确到秒)的String
 * @return string timeString
 */
string getTimeString_ICI(){
    struct tm *myTimeStruct;
    time_t myTime;
    myTime = time(NULL);
    myTimeStruct = localtime(&myTime);

    //string timeString = NULL;
    string timeString = to_string(myTimeStruct->tm_year + 1900) + to_string(myTimeStruct->tm_mon + 1) + to_string(myTimeStruct->tm_mday) + to_string(myTimeStruct->tm_hour)
                        + to_string(myTimeStruct->tm_min) + to_string(myTimeStruct->tm_sec);
    return timeString;
}

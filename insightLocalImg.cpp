//
// Created by Bosen on 28/11/2019.
// insight图传的本地图像测试方法
//

#include "insightLocalImg.h"

void imageLocal_insight(vector<Mat> & imageCatchesVector,string imgListPath,string imgFileFolder){
    /**
  * Part1第一次读取图像 根据预先设定的图像畸变参数对connex数传采集的图像进行预处理
  * Q: 图像预处理部分只做了畸变处理并去除connex黑边 可以根据需要再添加
  */

    //图像输入
    ifstream fileInputstream(imgListPath);
    string imgNameString;
    getline(fileInputstream, imgNameString);
    cout << imgNameString << endl;
    Mat imgCaptureMat = imread(imgNameString);

    if(imgCaptureMat.rows == INSIGHT_IN_HEI && imgCaptureMat.cols == INSIGHT_IN_WID){
        cout << ">>>> image capture success" << imgCaptureMat.size() << endl;
    }else{cout << "!!!!! image capture wrong size" << imgCaptureMat.size() << endl;}

    //图像size
    Size imgSize;
    imgSize = imgCaptureMat.size();

    // 设置相机畸变参数
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 1.1103660484579314e+03;
    cameraMatrix.at<double>(0, 1) = 0;
    cameraMatrix.at<double>(0, 2) = 640;
    cameraMatrix.at<double>(1, 1) = 1.1103660484579314e+03;
    cameraMatrix.at<double>(1, 2) = 360;

    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -3.1915717708126401e-01;
    distCoeffs.at<double>(1, 0) = 2.4807856799275141e-01;
    distCoeffs.at<double>(2, 0) = 0;
    distCoeffs.at<double>(3, 0) = 0;
    distCoeffs.at<double>(4, 0) = -2.6705206827354411e-01;

    //畸变map计算
    Mat imgMap1, imgMap2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 1, imgSize, 0),
                            imgSize, CV_16SC2, imgMap1, imgMap2);

    Mat imgCalMat(INSIGHT_IN_HEI, INSIGHT_IN_WID, imgCaptureMat.type()); //畸变校正后的图像
    Mat imgCalRezMat(INSIGHT_CAL_HEI, INSIGHT_CAL_WID, imgCaptureMat.type()); //畸变校正之后黑边去除的图像

    //畸变校正
    remap(imgCaptureMat, imgCalMat, imgMap1, imgMap2, INTER_LINEAR);
    //去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
    resizeImage(imgCalMat,imgCalRezMat,INSIGHT_CAL_REZ_TOP,INSIGHT_CAL_REZ_LEFT,INSIGHT_CAL_REZ_HEI,INSIGHT_CAL_REZ_WID);

    //将照片旋转成为竖幅便于拼接
    transpose(imgCalRezMat, imgCalRezMat);
    flip(imgCalRezMat,imgCalRezMat,1);

    cout << ">>>> frame 0 calibration success" << endl;
    cout << imgCalRezMat.rows << imgCalRezMat.cols << endl;
    if(imgCalRezMat.rows == INSIGHT_CAL_WID && imgCalRezMat.cols == INSIGHT_CAL_HEI){
        cout << ">>>> frame 0 calibration success" << imgCaptureMat.size() << endl;
    }else{cout << "!!!!! frame 0 calibration wrong size" << imgCaptureMat.size()  << endl;}

    string result_name = imgFileFolder + to_string(0) +"_"+ getTimeString_ILI() + ".jpg";

    //将该帧图像加入到image_catches Vector中
    imageCatchesVector.push_back(imgCalRezMat);

    /**
     * Part2重复读取图像 每次读取图像后都对图像进行去畸变预处理
     * Q: 实际飞行过程中会依据MAV_IMAGE_CAPTURED对图像进行处理
     */
    int i = 1;
    while(getline(fileInputstream, imgNameString))
    {

        Mat imgCaptureMat = imread(imgNameString);
        Mat imgCalMat(INSIGHT_IN_HEI, INSIGHT_IN_WID, imgCaptureMat.type()); //畸变校正后的图像
        Mat imgCalRezMat(INSIGHT_CAL_HEI, INSIGHT_CAL_WID, imgCaptureMat.type()); //畸变校正之后黑边去除的图像

        //畸变校正
        remap(imgCaptureMat, imgCalMat, imgMap1, imgMap2, INTER_LINEAR);
        //去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
        resizeImage(imgCalMat,imgCalRezMat,INSIGHT_CAL_REZ_TOP,INSIGHT_CAL_REZ_LEFT,INSIGHT_CAL_REZ_HEI,INSIGHT_CAL_REZ_WID);

        cout << ">>>> frame "<< i << " capture success" << imgCaptureMat.size() << endl;

        //将照片旋转成为竖幅便于拼接
        transpose(imgCalRezMat, imgCalRezMat);
        flip(imgCalRezMat,imgCalRezMat,1);

        if(imgCalRezMat.rows == INSIGHT_CAL_WID && imgCalRezMat.cols == INSIGHT_CAL_HEI){
            cout << ">>>> frame "<< i << " calibration success" << imgCalRezMat.size() << endl;
        }else{cout << "!!!!! frame "<< i << " calibration wrong size" << imgCalRezMat.size() << endl;}

        //将该帧图像加入到imageCatchesVector中
        imageCatchesVector.push_back(imgCalRezMat);
        //将每一帧图像保存在本机
        string result_name = imgFileFolder + to_string(i) +"_"+ getTimeString_ILI() + ".jpg";
        //char result_name[100];
        //string s = getTimeString();
        //char ss[14];
        //strcpy(ss,s.c_str());
        //cout << ss << endl;
        //sprintf(result_name, "%s%d%s%s%s", IMG_CAL_NAME, i, "_", ss  ,".jpg");
        imwrite(result_name, imgCalRezMat);
        cout << "image catches vector size = " << imageCatchesVector.size() << endl;
        i++;
    }
}

/**
 * 返回当前时间(精确到秒)的String
 * @return string timeString
 */
string getTimeString_ILI(){
    struct tm *myTimeStruct;
    time_t myTime;
    myTime = time(NULL);
    myTimeStruct = localtime(&myTime);

    //string timeString = NULL;
    string timeString = to_string(myTimeStruct->tm_year + 1900) + to_string(myTimeStruct->tm_mon + 1) + to_string(myTimeStruct->tm_mday) + to_string(myTimeStruct->tm_hour)
                        + to_string(myTimeStruct->tm_min) + to_string(myTimeStruct->tm_sec);
    return timeString;
}
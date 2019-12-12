//
// Created by Bosen on 28/11/2019.
// 处理未经预处理的amimon connex图传采集的图像
//

#include "amimonLocalImg.h"

/***
 * 采集由amimon connex的本地图像
 * @param imageCatchesVector
 * @param imgListPath
 * @param imgFileFolder
 */
void imageLocal_amiomon(vector<Mat> & imageCatchesVector,string imgListPath,string imgFileFolder) {
    /**
     * Part1第一次读取图像 根据预先设定的图像畸变参数对connex数传采集的图像进行预处理
     * Q: 图像预处理部分只做了畸变处理并去除connex黑边 可以根据需要再添加
     */

    //图像输入
    ifstream fileInputstream(imgListPath);
    string imgNameString;
    getline(fileInputstream, imgNameString);
    Mat imgCaptureMat = imread(imgNameString);

    if(imgCaptureMat.rows == CONNEX_IN_HEI && imgCaptureMat.cols == CONNEX_IN_WID){
        cout << ">>>> image capture success" << endl;
    }else{cout << "!!!!! image capture wrong size" << endl;}

    //图像size
    Size imgSize;
    imgSize = imgCaptureMat.size();

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

    Mat imgRezMat(CONNEX_IN_HEI, CONNEX_IN_WID, imgCaptureMat.type()); //去除connex图传上下黑边后的图像Mat
    Mat imgRezCalMat(CONNEX_IN_HEI, CONNEX_IN_WID, imgCaptureMat.type()); //畸变校正后的图像
    Mat imgRezCalRezMat(CONNEX_CAL_HEI, CONNEX_CAL_WID, imgCaptureMat.type()); //畸变校正之后黑边去除的图像

    //去除connex图传上下黑边_黑边只在高度上出现_高度上可用像素159_919
    resizeImage(imgCaptureMat,imgRezMat, CONNEX_REZ_TOP, CONNEX_REZ_LEFT, CONNEX_REZ_HEI, CONNEX_REZ_WID);
    //畸变校正
    remap(imgRezMat, imgRezCalMat, imgMap1, imgMap2, INTER_LINEAR);
    //去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
    resizeImage(imgRezCalMat,imgRezCalRezMat,CONNEX_CAL_REZ_TOP,CONNEX_CAL_REZ_LEFT,CONNEX_CAL_REZ_HEI,CONNEX_CAL_REZ_WID);

    //将照片旋转成为竖幅便于拼接
    transpose(imgRezCalRezMat, imgRezCalRezMat);
    flip(imgRezCalRezMat,imgRezCalRezMat,1);

    cout << ">>>> frame 0 calibration success" << endl;
    cout << imgRezCalRezMat.rows << imgRezCalRezMat.cols << endl;
    if(imgRezCalRezMat.rows == CONNEX_CAL_WID && imgRezCalRezMat.cols == CONNEX_CAL_HEI){
        cout << ">>>> frame 0 calibration success" << endl;
    }else{cout << "!!!!! frame 0 calibration wrong size" << endl;}

    string result_name = imgFileFolder + to_string(0) +"_"+ getTimeString_ALI() + ".jpg";

    //将该帧图像加入到image_catches Vector中
    imageCatchesVector.push_back(imgRezCalRezMat);

    /**
     * Part2重复读取图像 每次读取图像后都对图像进行去畸变预处理
     * Q: 实际飞行过程中会依据MAV_IMAGE_CAPTURED对图像进行处理
     */
    int i = 1;
    while(getline(fileInputstream, imgNameString))
    {

        Mat imgCaptureMat = imread(imgNameString);
        Mat imgRezMat(CONNEX_IN_HEI, CONNEX_IN_WID, imgCaptureMat.type()); //去除connex图传上下黑边后的图像Mat
        Mat imgRezCalMat(CONNEX_IN_HEI, CONNEX_IN_WID, imgCaptureMat.type()); //畸变校正后的图像
        Mat imgRezCalRezMat(CONNEX_CAL_HEI, CONNEX_CAL_WID, imgCaptureMat.type()); //畸变校正之后黑边去除的图像

        //去除connex图传上下黑边_黑边只在高度上出现_高度上可用像素159_919
        resizeImage(imgCaptureMat,imgRezMat, 159, 0, 760, 1920);
        //畸变校正
        remap(imgRezMat, imgRezCalMat, imgMap1, imgMap2, INTER_LINEAR);
        //去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
        resizeImage(imgRezCalMat,imgRezCalRezMat,50,25,980,1870);
        cout << ">>>> frame "<< i << " capture success" << endl;

        //将照片旋转成为竖幅便于拼接
        transpose(imgRezCalRezMat, imgRezCalRezMat);
        flip(imgRezCalRezMat,imgRezCalRezMat,1);

        if(imgRezCalRezMat.rows == 1870 && imgRezCalRezMat.cols == 980){
            cout << ">>>> frame "<< i << " calibration success" << endl;
        }else{cout << "!!!!! frame "<< i << " calibration wrong size" << endl;}

        //将该帧图像加入到imageCatchesVector中
        imageCatchesVector.push_back(imgRezCalRezMat);
        //将每一帧图像保存在本机
        string result_name = imgFileFolder + to_string(i) +"_"+ getTimeString_ALI() + ".jpg";
        imwrite(result_name, imgRezCalRezMat);
        i++;
    }
}

    /**处理无需去畸变的图像
    int i = 1;
    while (getline(fileInputstream, imgNameString)) {

            Mat imgCaptureMat = imread(imgNameString);
            string result_name = imgFileFolder + to_string(i) + "_" + getTimeString_ALI() + ".jpg";
            imageCatchesVector.push_back(imgCaptureMat);
            imwrite(result_name, imgCaptureMat);
            cout << i << endl;
            i++;
    }

}**/

/**
 * 返回当前时间(精确到秒)的String
 * @return string timeString
 */
string getTimeString_ALI(){
    struct tm *myTimeStruct;
    time_t myTime;
    myTime = time(NULL);
    myTimeStruct = localtime(&myTime);

    //string timeString = NULL;
    string timeString = to_string(myTimeStruct->tm_year + 1900) + to_string(myTimeStruct->tm_mon + 1) + to_string(myTimeStruct->tm_mday) + to_string(myTimeStruct->tm_hour)
                        + to_string(myTimeStruct->tm_min) + to_string(myTimeStruct->tm_sec);
    return timeString;
}

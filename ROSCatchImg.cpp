//
// Created by Bosen on 2/12/2019.
// 本代码需要配合ROS使用，ROS通过读取航路点坐标及飞机实时位置，在指定航路点处拍照并将拍到的照片放到本地某文件夹下
// 因为愚蠢的规则，在这个函数下进行了单张图像的yolo检测及圆检测，如果之后程序出现问题还可以拿单张图像检测分
//

#include "ROSCatchImg.h"
#include "darknetProcess.h"
#include "circleProcess.h"

#define SINGLE_YOLO_DETECT_FOLDER "/home/ubuntu/Project/2019Match_ZSKT/Result/singleyoloresult/"     //单张yolo检测结果图存放位置
#define YOLO_CFG_PATH "/home/ubuntu/Project/2019Match_ZSKT/Result/yolomodels/yolov31203-voc.cfg"    //yolo检测的cfg文件存放位置
#define YOLO_WEG_PATH2 "/home/ubuntu/git/darknet-master/backup/yolov31203-voc_final.weights"        //yolo检测的weight文件存放位置
#define YOLO_NAM_PATH "/home/ubuntu/Project/2019Match_ZSKT/Result/yolomodels/coco.names"            //yolo检测的name文件存放位置

//每个检测到的target结果Class
struct this_detect_result_class{
    int classId; //class id 类序号
    string label; //class label 类标识
    float confidences; //confidences 概率
    int left = -1; //左上角点x坐标(在整个大图中的坐标)
    int top = -1; //左上角点y坐标(在整个大图中的坐标)
    int width = -1; //宽度
    int height = -1; //高度
    float center_distance = -1; //距中心圆的距离in meter
};



void imageCatch_ROSinsight(vector<Mat> & imageCatchesVector, string imgInFolder, string imgOutFolder, int imgTotalNum) {

    string modelConfiguration = YOLO_CFG_PATH;
    string modelWeights = YOLO_WEG_PATH2;
    string classesFile = YOLO_NAM_PATH;

    Mat imgCaptureMat;      //当前的capture mat
    string thisInImgName;   //当前读入的的image名称
    int thisImgNum = 0;     //当前采集的图像数量
    struct stat buffer;     //文件stat buffer

    thisInImgName = imgInFolder + to_string(thisImgNum) + ".jpg";

    /**
     * 依据文件夹中的jpg图像循环将图像存入缓冲区
     */
    for (; thisImgNum < imgTotalNum;) {
        cout << ">>>> Try find image:" << thisInImgName << endl;

        while (1) {
            ///如果查询文件夹发现jpg文件
            if (stat(thisInImgName.c_str(), &buffer) == 0) {
                waitKey(200); ///等待200ms

                ///从img file中读入图像
                imgCaptureMat = imread(thisInImgName);

                //imshow("imgCatch", imgCatchMat);
                Mat imgCaptureRemapMat;
                imageReMap(imgCaptureMat, imgCaptureRemapMat);

                ///将照片顺时针旋转成为竖幅便于拼接
                transpose(imgCaptureRemapMat, imgCaptureRemapMat);
                flip(imgCaptureRemapMat, imgCaptureRemapMat, 1);

                ///::比赛第四轮使用的部分::
                ///图像预处理：将图像转换为YCRCB空间对Y通道进行直方图均衡预处理
                ///此处对单张图像进行了色彩空间预处理，更多考虑的是在结果呈现时不要多个组给裁判看同一种结果，其实并不需要这一步
                /**
                Mat imgYCrCb;
                cvtColor(imgCaptureRemapMat, imgYCrCb, COLOR_BGR2HSV);
                std::vector<Mat> channels;
                split(imgYCrCb, channels);  ///色彩空间分离
                equalizeHist(channels[2],channels[2]);  ///直方图均衡
                merge(channels, imgCaptureRemapMat);    ///色彩空间返回
                cvtColor(imgCaptureRemapMat,imgCaptureRemapMat,COLOR_HSV2BGR);
                **/

                ///将该帧图像加入到image_catches Vector中
                imageCatchesVector.push_back(imgCaptureRemapMat);

                ///将采集到的图像保存到本机
                string imgFileName = imgOutFolder + to_string(thisImgNum) + "_" + getTimeString_RCI() + ".jpg";
                imwrite(imgFileName, imgCaptureRemapMat);
                cout << ">>>> Catch Image: " << thisImgNum << "success" << endl;


                /*
                cout << "进入单张yolo检测:  "  << thisImgNum << endl;
                //该张图片检测的目标信息vector
                vector<detect_class> singleYoloDetectClassVector;
                //yolo检测后的图像（如果需要对单张yolo检测结果写标签可以输出这个Mat
                Mat imageSingleYoloMat;
                Mat imagecloneYoloMat;
                imagecloneYoloMat = imgCaptureRemapMat.clone();
                //执行yolo检测
                detect_image(imagecloneYoloMat, imageSingleYoloMat, singleYoloDetectClassVector, modelWeights, modelConfiguration, classesFile);
                //将该张图片的yoloresult pushback到yoloResultImgVector
                string thisNameString = SINGLE_YOLO_DETECT_FOLDER  + to_string(thisImgNum) + ".jpg";
                imwrite(thisNameString,imageSingleYoloMat);
*/
                cout << ">>>> Do single yolo detect: "  << thisImgNum << endl;
                //该张图片检测的目标信息vector
                vector<detect_class> singleYoloDetectClassVector;
                ///yolo检测后的图像结果（这个Mat里面有yolo检测结果的bounding box，如果需要可以输出
                Mat imageSingleYoloMat;
                ///clone一份原始CaptureRemapMat作为yolo检测的源图
                Mat imagecloneYoloMat;
                imagecloneYoloMat = imgCaptureRemapMat.clone();
                transpose(imagecloneYoloMat, imagecloneYoloMat);
                flip(imagecloneYoloMat, imagecloneYoloMat, 0);
                ///执行yolo检测
                detect_image(imagecloneYoloMat, imageSingleYoloMat, singleYoloDetectClassVector, modelWeights, modelConfiguration, classesFile);
                //将该张图片的yoloresult pushback到yoloResultImgVector

                ///clone一份原始CaptureRemapMat作为圆检测的源图
                Mat imageSingleCircleMat;
                imageSingleCircleMat = imgCaptureRemapMat.clone();
                transpose(imageSingleCircleMat, imageSingleCircleMat);
                flip(imageSingleCircleMat, imageSingleCircleMat, 0);

                vector<circle_class> circleVector; ///circleProcess检测维持的一个圆序列
                Point2f thisPoint;
                thisPoint.x = 0.0;
                thisPoint.y = 0.0;
                detectCircle(imageSingleCircleMat,circleVector,thisPoint,thisImgNum); ///执行圆检测


                ///输出圆检测结果、在图上显示bounding box、计算距离
                if(circleVector.size() > 0){
                    int possibleCircleCenterX = 0; //结果圆心x坐标
                    int possibleCircleCenterY = 0; //结果圆心y坐标
                    float possibleCircleRadius = 0; //结果圆半径
                    if(circleVector.size() != 0){
                        possibleCircleCenterX = circleVector[0].left;
                        possibleCircleCenterY = circleVector[0].top;
                        possibleCircleRadius = circleVector[0].radius;
                    }

                    for(int i = 0; i < singleYoloDetectClassVector.size(); i++){
                        int singleCircleDistance = sqrt(pow((singleYoloDetectClassVector[i].left - possibleCircleCenterX),2) +
                                                        pow((singleYoloDetectClassVector[i].top - possibleCircleCenterY),2)) * (1 / possibleCircleRadius);
                        drawPred(singleYoloDetectClassVector[i].classId, singleCircleDistance, singleYoloDetectClassVector[i].left, singleYoloDetectClassVector[i].top
                                , (singleYoloDetectClassVector[i].left + singleYoloDetectClassVector[i].width), (singleYoloDetectClassVector[i].top + singleYoloDetectClassVector[i].height), imageSingleCircleMat);
                    }
                }else{
                    for(int i = 0; i < singleYoloDetectClassVector.size(); i++){
                        drawPred(singleYoloDetectClassVector[i].classId,0, singleYoloDetectClassVector[i].left, singleYoloDetectClassVector[i].top
                                , (singleYoloDetectClassVector[i].left + singleYoloDetectClassVector[i].width), (singleYoloDetectClassVector[i].top + singleYoloDetectClassVector[i].height), imageSingleCircleMat);
                    }
                }

                ///将单张检测结果保存在本地
                string thisNameString = SINGLE_YOLO_DETECT_FOLDER  + to_string(thisImgNum) + ".jpg";
                transpose(imageSingleCircleMat, imageSingleCircleMat);
                flip(imageSingleCircleMat, imageSingleCircleMat, 1);
                imwrite(thisNameString,imageSingleCircleMat);

                break;
            }
        }
        thisImgNum = thisImgNum + 1;
        thisInImgName = imgInFolder + to_string(thisImgNum) + ".jpg";
    }

}

/***
 * 进行图像畸变矫正
 */
void imageReMap(Mat & srcMat, Mat & distMat) {
    //图像size
    Size imgSize;
    imgSize = srcMat.size();

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

    ///畸变map计算
    Mat imgMap1, imgMap2;
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                            getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 1, imgSize, 0),
                            imgSize, CV_16SC2, imgMap1, imgMap2);

    Mat imgCalMat(INSIGHT_IN_HEI, INSIGHT_IN_WID, srcMat.type()); //畸变校正后的图像
    Mat imgCalRezMat(INSIGHT_CAL_HEI, INSIGHT_CAL_WID, srcMat.type()); //畸变校正之后黑边去除的图像

    ///畸变校正
    remap(srcMat, imgCalMat, imgMap1, imgMap2, INTER_LINEAR);
    ///去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
    resizeImage(imgCalMat,imgCalRezMat,INSIGHT_CAL_REZ_TOP,INSIGHT_CAL_REZ_LEFT,INSIGHT_CAL_REZ_HEI,INSIGHT_CAL_REZ_WID);

    distMat = imgCalRezMat.clone();
    cout << ">>>> remap this image success" << endl;
}

string getTimeString_RCI(){
    struct tm *myTimeStruct;
    time_t myTime;
    myTime = time(NULL);
    myTimeStruct = localtime(&myTime);

    //string timeString = NULL;
    string timeString = to_string(myTimeStruct->tm_year + 1900) + to_string(myTimeStruct->tm_mon + 1) + to_string(myTimeStruct->tm_mday) + to_string(myTimeStruct->tm_hour)
                        + to_string(myTimeStruct->tm_min) + to_string(myTimeStruct->tm_sec);
    return timeString;
}

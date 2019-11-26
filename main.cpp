
#include <iostream>

#include <darknet.h>
#include "darknetProcess.h"
#include "imageProcess.h"
#include "imageCut.h"
#include "time.h"
#include "circleProcess.h"

#define TEST_IMG_PATH "/home/ubuntu/git/CatchAndDetect/img.txt"
#define YOLO_CFG_PATH "/home/ubuntu/git/darknet-master/cfg/yolov3test1-voc.cfg"
#define YOLO_WEG_PATH "/home/ubuntu/git/darknet-master/backup/yolov3tuchuan-voc_10000.weights"
#define YOLO_NAM_PATH "/home/ubuntu/git/darknet-master/data/coco.names"
#define IMG_CAL_NAME "/home/ubuntu/git/CatchAndDetect/calibration/"
#define IMG_STI_NAME "/home/ubuntu/git/CatchAndDetect/stitcher/"
#define IMG_YOLO_NAME "/home/ubuntu/git/CatchAndDetect/yoloresult/"

#define CONNEX_IN_HEI 1080 //CONNEX图传传入的图像_高度
#define CONNEX_IN_WID 1920 //CONNEX图传传入的图像_宽度
#define CONNEX_CAL_HEI 980 //CONNEX图传校正后的图像_高度
#define CONNEX_CAL_WID 1870 //CONNEX图传校正后的图像_宽度

//去除connex图传上下黑边_黑边只在高度上出现_高度上可用像素159_919
#define CONNEX_REZ_TOP 159
#define CONNEX_REZ_LEFT 0
#define CONNEX_REZ_HEI 760
#define CONNEX_REZ_WID 1920

//去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
#define CONNEX_CAL_REZ_TOP 50
#define CONNEX_CAL_REZ_LEFT 25
#define CONNEX_CAL_REZ_HEI 980
#define CONNEX_CAL_REZ_WID 1870

//是否在stitch中使用GPU
bool try_use_gpu = true;
//采集的所有图像Vector
vector<Mat> imageCatchesVector;
//stitcherCut的所有图像Vector
vector<Mat> imageStitchersVector;

//函数定义 当前时间的string
string getTimeString();

//yolo检测的每张图像结果Class
struct yolo_result_class{
    int imagenum; //图像序号
    vector<detect_class> detectclass; //图像中识别出的类vector
    int width; //图像宽度
    int height; //图像高度
    Point2f point_in_sitich; //在拼接原图上左上点的坐标
};

//每个检测到的target结果Class
struct detect_result_class{
    int classId; //class id 类序号
    string label; //class label 类标识
    float confidences; //confidences 概率
    int left = -1; //左上角点x坐标(在整个大图中的坐标)
    int top = -1; //左上角点y坐标(在整个大图中的坐标)
    int width = -1; //宽度
    int height = -1; //高度
    float center_distance = -1; //距中心圆的距离in meter
};

//检测到的Circle结果Class
struct circle_result_class{
    float left; //圆心x坐标(在整个大图中的坐标)
    float top; //圆心y坐标(在整个大图中的坐标)
    float radius; // 半径(以像素点表示)
};

int main()
{
    /**
     * 定义全局保存变量
     */
    Mat stitcherResultMat; //拼接结果图像Mat
    Mat yoloResultMat; //yolo检测结果图像Mat

    vector<detect_result_class> detectResultVector; //识别目标的结果Vector
    circle_result_class circleResult; //识别圆的结果

    /**
     * Part1第一次读取图像 根据预先设定的图像畸变参数对connex数传采集的图像进行预处理
     * Q: 图像预处理部分只做了畸变处理并去除connex黑边 可以根据需要再添加
     */

    //图像输入
    ifstream fileInputstream(TEST_IMG_PATH);
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
        string result_name = IMG_CAL_NAME + to_string(i) +"_"+ getTimeString() + ".jpg";
        //char result_name[100];
        //string s = getTimeString();
        //char ss[14];
        //strcpy(ss,s.c_str());
        //cout << ss << endl;
        //sprintf(result_name, "%s%d%s%s%s", IMG_CAL_NAME, i, "_", ss  ,".jpg");
        imwrite(result_name, imgRezCalRezMat);
        i++;
    }
    /**
     * Part3运行Stitcher类进行图像拼接
     * Q: 是否需要求改Stitcher类？虽然现在看来是不需要的，默认参数也足够使用
     */
    //Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
    //建立stitcher类
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS,try_use_gpu);
    //建立stitcher status类
    Stitcher::Status status = stitcher->stitch(imageCatchesVector, stitcherResultMat);
    if (status != Stitcher::OK){
        cout << "!!!! Can't stitch images, error code = " << status << endl;
    }else{
        cout << ">>>> Stitch images success" << endl;
    }
    cout << ">>>> Stitch images count: " << imageCatchesVector.size() << endl;

    //将Stitcher结果写到本地
    string resultNameString = IMG_STI_NAME + getTimeString() + ".jpg";
    //char resultNameString[100];
    //string s = getTimeString();
    //char ss[14];
    //strcpy(ss,s.c_str());
    //sprintf(resultNameString, "%s%s%s", IMG_CAL_NAME, ss ,".jpg");
    //namedWindow(resultNameString);
    //imshow(resultNameString,stitcherResultMat);
    imwrite(resultNameString,stitcherResultMat);


    /**
     * Part4 剪切stitch图像以供yolo使用
     *
     */
    vector<yolo_image> yoloSrcImgVector;
    imageCut(stitcherResultMat, yoloSrcImgVector);
    if(yoloSrcImgVector.size() != 0){
        cout << ">>>> Cut stitch images success" << endl;
    } else{
        cout << "!!!! Cut stitch images failure" << endl;
    }

    /**
    * Part5 调用darknet类并执行yolo进行目标检测
    *
    */
    // Give the configuration and weight files for the model
    // 读取模型文件，请自行修改相应路径
    string modelConfiguration = YOLO_CFG_PATH;
    string modelWeights = YOLO_WEG_PATH;
    string classesFile = YOLO_NAM_PATH;

    //每张图片的yolo检测结果Vector
    vector<yolo_result_class> yoloResultImgVector;

    //循环检测每yoloSrcImgVector的yolo_image 并把结果pushback到yoloResultImgVector
    for(int i = 0; i < yoloSrcImgVector.size(); i++){
        //该张图片检测的目标信息vector
        vector<detect_class> thisDetectVector;
        //yolo检测后的图像（如果需要对单张yolo检测结果写标签可以输出这个Mat
        Mat dstImageMat;

        //执行yolo检测
        detect_image(yoloSrcImgVector[i].yolo_image, dstImageMat, thisDetectVector,  modelWeights, modelConfiguration, classesFile);

        //该张图片的yoloresult
        yolo_result_class thisYoloResult;
        thisYoloResult.imagenum = i;
        thisYoloResult.detectclass.swap(thisDetectVector);
        thisYoloResult.height = yoloSrcImgVector[i].height;
        thisYoloResult.width = yoloSrcImgVector[i].width;
        thisYoloResult.point_in_sitich = yoloSrcImgVector[i].point_in_sitich;

        //将该张图片的yoloresult pushback到yoloResultImgVector
        yoloResultImgVector.push_back(thisYoloResult);

        //清理thisDetectVector
        thisDetectVector.clear();
    }

    if(yoloResultImgVector.size() == yoloSrcImgVector.size()){
        cout << ">>>> YOLO images success" << endl;
        cout << ">>>> YOLO images count: " << yoloResultImgVector.size() << endl;
    } else{
        cout << "!!!! YOLO images failure" << endl;
    }

    /***
     * Part6 调用circleProcess执行Hough圆检测
     */
    //执行hough circle检测
    vector<circle_class> circleVector; //circleProcess检测维持的一个圆序列
    //测试圆 Mat yoloimage = imread("/home/zhouhelu/CLionProjects/CatchAndDetect/stitcher/yuan.jpg");
    //detectCircle(yoloimage,circleVector,yoloSrcImgVector[0].point_in_sitich);
    for(int i = 0; i < yoloSrcImgVector.size(); i++){
        detectCircle(yoloSrcImgVector[i].yolo_image,circleVector,yoloSrcImgVector[i].point_in_sitich);
    }


    if(circleVector.size() != 0){
        cout << ">>>> circle detect success" << endl;
        int possibleCircleCenterX; //结果圆心x坐标
        int possibleCircleCenterY; //结果圆心y坐标
        int possibleCircleRadius; //结果圆半径
        //cout << ">>>> circle radius" << circleVector[0].left <<  "circle x" << circleVector[0].top << "circle y" << circleVector[0].radius << endl;
        for(int i = 0; i < circleVector.size(); i++){
            possibleCircleCenterX = possibleCircleCenterX + circleVector[i].left;
            possibleCircleCenterY = possibleCircleCenterY + circleVector[i].top;
            possibleCircleRadius = possibleCircleRadius + circleVector[i].radius;
        }

        possibleCircleCenterX = possibleCircleCenterX / circleVector.size();
        possibleCircleCenterY = possibleCircleCenterY / circleVector.size();
        possibleCircleRadius = possibleCircleRadius /  circleVector.size();

        //将结果输出到circleResult中
        circleResult.radius = possibleCircleRadius;
        circleResult.left = possibleCircleCenterX;
        circleResult.top = possibleCircleCenterY;

        cout << ">>>> circle radius" << circleResult.radius <<  "circle x" << circleResult.left << "circle y" << circleResult.top << endl;

    } else{
        cout << "!!!! circle detect failure" << endl;
    }

    //使用平均值的方法减少圆检测的误差

    /***
     * Part7 对于所有的yoloyresultimage遍历 将检测结果detectresult转化到大图坐标系下
     */
    for(int i = 0; i < yoloResultImgVector.size(); i++){
        for(int j= 0; j < yoloResultImgVector[i].detectclass.size();j++){
            detect_result_class thisDetectResult;
            thisDetectResult.classId = yoloResultImgVector[i].detectclass[j].classId;
            thisDetectResult.label = yoloResultImgVector[i].detectclass[j].label;
            thisDetectResult.confidences = yoloResultImgVector[i].detectclass[j].confidences;
            thisDetectResult.width = yoloResultImgVector[i].detectclass[j].width;
            thisDetectResult.height = yoloResultImgVector[i].detectclass[j].height;
            thisDetectResult.left = yoloResultImgVector[i].detectclass[j].left + yoloResultImgVector[i].point_in_sitich.x;
            thisDetectResult.top = yoloResultImgVector[i].detectclass[j].top + yoloResultImgVector[i].point_in_sitich.y;

            //利用圆的半径计算世界坐标系下 目标到中心圆的距离
            thisDetectResult.center_distance = sqrt(pow((thisDetectResult.left - circleResult.left),2) + pow((thisDetectResult.top - circleResult.top),2))
                    * circleResult.radius;

            //将目标检测result push back到全局result vector中
            detectResultVector.push_back(thisDetectResult);
        }
    }

    if(detectResultVector.size() != 0){
        cout << ">>>> Detect target success" << endl;
        cout << ">>>> Detect target count: " << detectResultVector.size() << endl;
    } else{
        cout << "!!!! Detect target failure" << endl;
    }


    /***
    * Part8 在大图中绘制bounding box
    */
    //将大图stitcherResultMat clone 到yoloResultMat下
    yoloResultMat = stitcherResultMat.clone();
    //绘制
    for(int i = 0; i < detectResultVector.size(); i++){
        drawPred(detectResultVector[i].classId, detectResultVector[i].confidences, detectResultVector[i].left, detectResultVector[i].top
                , (detectResultVector[i].left + detectResultVector[i].width), (detectResultVector[i].top + detectResultVector[i].height), yoloResultMat);
    }

    cout << ">>>> Bounding box Print success" << endl;

    //写图像
    imwrite(IMG_YOLO_NAME + getTimeString() + ".jpg" ,yoloResultMat);
    cout << ">>>> Final Image wirte success" << endl;

    //cv::waitKey(0);
    return 0;
}

/**
 * 返回当前时间(精确到秒)的String
 * @return string timeString
 */
string getTimeString(){
    struct tm *myTimeStruct;
    time_t myTime;
    myTime = time(NULL);
    myTimeStruct = localtime(&myTime);

    //string timeString = NULL;
    string timeString = to_string(myTimeStruct->tm_year + 1900) + to_string(myTimeStruct->tm_mon + 1) + to_string(myTimeStruct->tm_mday) + to_string(myTimeStruct->tm_hour)
            + to_string(myTimeStruct->tm_min) + to_string(myTimeStruct->tm_sec);
    return timeString;
}

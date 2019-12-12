
#include <iostream>

#include <darknet.h>
#include "darknetProcess.h"
#include "imageProcess.h"
#include "imageCut.h"
#include "time.h"
#include "circleProcess.h"
#include "amimonLocalImg.h"
#include "insightCatchImg.h"
#include "ROSCatchImg.h"
#include "imagePreProcess.h"
#include "insightLocalImg.cpp"
#include "imageForceStitch.h"

#define TEST_IMG_PATH "/home/ubuntu/Project/2019Match_ZSKT/CatchAndDetect_local/img.txt"            //local步骤中image list的存放位置
#define YOLO_CFG_PATH "/home/ubuntu/Project/2019Match_ZSKT/Result/yolomodels/yolov31203-voc.cfg"    //YOLO CFG文件的存放位置
#define YOLO_WEG_PATH "/home/ubuntu/git/darknet-master/backup/yolov31203-voc_final.weights"         //YOLO 权重文件的存放位置
#define YOLO_NAM_PATH "/home/ubuntu/Project/2019Match_ZSKT/Result/yolomodels/coco.names"            //YOLO类别文件的存放位置
#define IMG_CAL_NAME "/home/ubuntu/Project/2019Match_ZSKT/Result/calibration/"                      //采集并完成畸变校正的图像位置
#define SINGLE_YOLO_DETECT_FOLDER "/home/ubuntu/Project/2019Match_ZSKT/Result/singleyoloresult/"    //单张yolo检测结果存放位置
#define IMG_STI_NAME "/home/ubuntu/Project/2019Match_ZSKT/Result/stitcher/"                         //拼接完成的大图的位置
#define IMG_YOLO_NAME "/home/ubuntu/Project/2019Match_ZSKT/Result/yoloresult/"                      //yolo最终得到结果的大图的位置
#define RTSP_ADD "rtsp://192.168.2.220:554/stream/1"                                                //insight图传的rtsp地址
//#define QGC_TXT_PATH "/home/ubuntu/Project/2019Match_ZSKT/Result/QGCtxt/"                         //QGC使用Mavlink采集图像，保存的txt标志位的地址
#define ROS_JPG_PATH "/home/ubuntu/Project/2019Match_ZSKT/Result/catch/"                            //MAVROS采集图像，保存的JPG文件的地址

///由于使用的GOPRO图像在25米条件下拍摄的图像大小大约在30m*15m左右，一列图无法覆盖50*100区域，因此飞两遍，分开两行拼接
#define IMG_NUM_TOTAL 20     //图像采集的总数量
#define IMG_CAL 10        //一行图像的数量

#define CIRCLE_TRUE_METER 1 ///真实条件下圆的半径，用于做比例尺计算目标到圆的距离


///是否在stitch中使用GPU
bool try_use_gpu = true;
///采集的所有去除畸变图像Vector
vector<Mat> imageCatchesVector;

///图像增强之后的Vector
vector<Mat> imageEnhVector;

///进行单张yolo检测的图像Vector
vector<Mat> imageSingleYoloVector;


vector<Mat> imageCatchesVector1;
vector<Mat> imageCatchesVector2;
//vector<Mat> imageCatchesVector3;
vector<Mat> imageCatchesVectorall;


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

void getRepeatDetect(vector<detect_result_class> & detectVectorClass);

int main()
{
    /**
     * 定义全局保存变量
     */
    // Give the configuration and weight files for the model
    // 读取模型文件，请自行修改相应路径
    string modelConfiguration = YOLO_CFG_PATH;
    string modelWeights = YOLO_WEG_PATH;
    string classesFile = YOLO_NAM_PATH;

    Mat stitcherResultMat; //拼接结果图像Mat
    Mat yoloResultMat; //yolo检测结果图像Mat

    vector<detect_result_class> singleDetectVector; //仅去畸变后的yolo检测结果
    vector<detect_result_class> detectResultVector; //图像裁剪后识别目标的结果Vector
    circle_result_class circleResult; //识别圆的结果

    /**
     * Part1 读取图像 根据预先设定的图像畸变参数对connex数传采集的图像进行预处理
     * Q: 图像预处理部分只做了畸变处理并去除connex黑边 可以根据需要再添加
     */

    ///采集本地的由insight图传采集的图像
    //string imgListPath = TEST_IMG_PATH;
    //string imgFileFolder = IMG_CAL_NAME;
    //imageLocal_insight(imageCatchesVectorall, imgListPath, imgFileFolder);
    //cout << imageCatchesVectorall.size() << endl;

    ///采集由QGC标志位得到的insight图传采集的图像
    //string rtspAddress = RTSP_ADD;
    //string qgcTxtFolder = QGC_TXT_PATH;
    //string imgFileFolder = IMG_CAL_NAME;
    int imgTotalNum = IMG_NUM_TOTAL;
    //imageCatch_insight(imageCatchesVector,rtspAddress,qgcTxtFolder,imgFileFolder,imgTotalNum);

    ///采集由ROS得到的insight图传采集的图像
    string imgListPath = TEST_IMG_PATH;
    string imgFileFolder = IMG_CAL_NAME;
    //imageLocal_amiomon(imageCatchesVector,imgListPath, imgFileFolder);
    imageCatch_ROSinsight(imageCatchesVector,ROS_JPG_PATH,IMG_CAL_NAME, IMG_NUM_TOTAL);//去畸变
    imageForceStitch();

    /***
     * Part2 图像增强
     * 因为草地特征值不够，无法进行正常的stiticher类拼接，因此在此处加入图像增强算法，意图增加可供拼接的特征
     */
    ///图像增强
    //imageEqual2(imageCatchesVector, imagezqVector);


    /**
     * Part3运行Stitcher类进行图像拼接
     * Q: 是否需要求改Stitcher类？虽然现在看来是不需要的，默认参数也足够使用
     */

    ///imageCatchesVector的两行拼接区间拷贝
    //imageCatchesVector2.assign(imagezqVector.begin() + 2,imagezqVector.begin() + IMG_CAL-1);
    //imageCatchesVector1.assign(imagezqVector.begin() + IMG_CAL+1,imagezqVector.begin() + (IMG_CAL * 2-2));
    imageCatchesVector1.assign(imageCatchesVector.begin() ,imageCatchesVector.begin() + IMG_CAL);
    imageCatchesVector2.assign(imageCatchesVector.begin() + IMG_CAL,imageCatchesVector.begin() + (IMG_CAL * 2));
    cout << ">>>> First Col image total num: " << imageCatchesVector1.size() << endl;
    cout << ">>>> Secend Col image total num: " << imageCatchesVector2.size() << endl;

    ///拼接第一行
    string resultNameString = IMG_STI_NAME  + getTimeString() + "_col2" + ".jpg";
    Ptr<Stitcher> stitcher1 = Stitcher::create(Stitcher::SCANS,try_use_gpu);
    Stitcher::Status status1 = stitcher1->stitch(imageCatchesVector1, stitcherResultMat);
    imageCatchesVectorall.push_back(stitcherResultMat);
    resultNameString = IMG_STI_NAME + getTimeString() + "_col1" + ".jpg";
    imwrite(resultNameString,stitcherResultMat);
    if (status1 != Stitcher::OK){
        cout << "!!!! Col 1 Can't stitch images, error code = " << status1 << endl;
    }else{
        cout << ">>>> Stitch Col 1 images success" << endl;
    }

    ///两列图像的拼接，分别拼接第一行、第二行
    ///拼接第二行
    Ptr<Stitcher> stitcher2 = Stitcher::create(Stitcher::SCANS,try_use_gpu);
    Stitcher::Status status2 = stitcher2->stitch(imageCatchesVector1, stitcherResultMat);

    ///12.06.10:09 将第二行的图旋转两次（旋转180度）进行拼接
    transpose(stitcherResultMat, stitcherResultMat);
    flip(stitcherResultMat, stitcherResultMat, 1);
    transpose(stitcherResultMat, stitcherResultMat);
    flip(stitcherResultMat, stitcherResultMat, 1);

    imageCatchesVectorall.push_back(stitcherResultMat);

    imwrite(resultNameString,stitcherResultMat);
    if (status2 != Stitcher::OK){
        cout << "!!!! Col 2 Can't stitch images, error code = " << status2 << endl;
    }else{
        cout << ">>>> Stitch Col 2 images success" << endl;
    }

    //Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
    ////建立stitcher类，拼接第一行、第二行进行大图的拼接
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS, try_use_gpu);
    ///建立stitcher status类
    Stitcher::Status status = stitcher->stitch(imageCatchesVectorall, stitcherResultMat);
    if (status != Stitcher::OK){
        cout << "!!!! Final image Can't stitch, error code = " << status << endl;
    }else{
        cout << ">>>> Stitch final image success" << endl;
    }
    cout << ">>>> 增强后用的图像拼接数量: " << imageCatchesVector.size() << endl;

    ///将Stitcher结果写到本地
    resultNameString = IMG_STI_NAME + getTimeString() + ".jpg";
    imwrite(resultNameString,stitcherResultMat);
    ///将大图stitcherResultMat clone 到yoloResultMat下
    yoloResultMat = stitcherResultMat.clone();


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

    ///每张图片的yolo检测结果Vector
    vector<yolo_result_class> yoloResultImgVector;

    ///循环检测每yoloSrcImgVector的yolo_image 并把结果pushback到yoloResultImgVector
    for(int i = 0; i < yoloSrcImgVector.size(); i++){
        ///该张图片检测的目标信息vector
        vector<detect_class> thisDetectVector;
        ///yolo检测后的图像（如果需要对单张yolo检测结果写标签可以输出这个Mat
        Mat dstImageMat;

        ///执行yolo检测
        detect_image(yoloSrcImgVector[i].yolo_image, dstImageMat, thisDetectVector,  modelWeights, modelConfiguration, classesFile);

        ///该张图片的yoloresult
        yolo_result_class thisYoloResult;
        thisYoloResult.imagenum = i;
        thisYoloResult.detectclass.swap(thisDetectVector);
        thisYoloResult.height = yoloSrcImgVector[i].height;
        thisYoloResult.width = yoloSrcImgVector[i].width;
        thisYoloResult.point_in_sitich = yoloSrcImgVector[i].point_in_sitich;

        ///将该张图片的yoloresult pushback到yoloResultImgVector
        yoloResultImgVector.push_back(thisYoloResult);

        ///清理thisDetectVector
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

    Mat totalCircleDetectMat = yoloResultMat.clone();
    vector<circle_class> circleVector; //circleProcess检测维持的一个圆序列
    //测试圆 Mat yoloimage = imread("/home/zhouhelu/CLionProjects/CatchAndDetect/stitcher/yuan.jpg");
    //detectCircle(yoloimage,circleVector,yoloSrcImgVector[0].point_in_sitich);
    Point2f thisCirclePoint;
    thisCirclePoint.x = 0; thisCirclePoint.y = 0;
    detectCircle(totalCircleDetectMat,circleVector,thisCirclePoint,1);
    cout << ">>>> circle count  " << circleVector.size() << endl;

    if(circleVector.size() != 0){
        cout << ">>>> circle detect success" << endl;
        int possibleCircleCenterX = 0; //结果圆心x坐标
        int possibleCircleCenterY = 0; //结果圆心y坐标
        int possibleCircleRadius = 0; //结果圆半径
        //cout << ">>>> circle radius" << circleVector[0].left <<  "circle x" << circleVector[0].top << "circle y" << circleVector[0].radius << endl;
        for(int i = 0; i < circleVector.size(); i++){
            possibleCircleCenterX = possibleCircleCenterX + circleVector[i].left;
            possibleCircleCenterY = possibleCircleCenterY + circleVector[i].top;
            possibleCircleRadius = possibleCircleRadius + circleVector[i].radius;
        }

        possibleCircleCenterX = possibleCircleCenterX / circleVector.size();
        possibleCircleCenterY = possibleCircleCenterY / circleVector.size();
        possibleCircleRadius = possibleCircleRadius /  circleVector.size();

        ///将结果输出到circleResult中
        circleResult.radius = possibleCircleRadius;
        circleResult.left = possibleCircleCenterX;
        circleResult.top = possibleCircleCenterY;

        cout << ">>>> circle radius  " << circleResult.radius <<  "circle x  " << circleResult.left << "circle y  " << circleResult.top << endl;
        cout << ">>>> circle count  " << circleVector.size()<< endl;
        Point center(possibleCircleCenterX, possibleCircleCenterY);
        int radius = round(possibleCircleRadius);
        circle(yoloResultMat, center, radius, Scalar(255, 0, 0), 4, 4, 0);

    } else{
        cout << "!!!! circle detect failure" << endl;
    }

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

            ///利用圆的半径计算世界坐标系下 目标到中心圆的距离
            ///此处定义：圆的半径为1
            thisDetectResult.center_distance = sqrt(pow((thisDetectResult.left - circleResult.left),2) + pow((thisDetectResult.top - circleResult.top),2))
                    * (CIRCLE_TRUE_METER / circleResult.radius);

            ///将目标检测result push back到全局result vector中
            detectResultVector.push_back(thisDetectResult);
        }
    }

    /***
     * Part8 计算各个类别的数量，并打印到大图上
     */

    if(detectResultVector.size() != 0){
        cout << ">>>> Detect target success" << endl;
        cout << ">>>> Detect target count: " << detectResultVector.size() << endl;
    } else{
        cout << "!!!! Detect target failure" << endl;
    }

    ///定义三种目标的Result class vector
    vector<detect_result_class> planeResultClassVector;
    int planeResultNum;
    vector<detect_result_class> tankResultClassVector;
    int tankResultNum;
    vector<detect_result_class> shipResultClassVector;
    int shipResultNum;


    for(int i = 0; i < detectResultVector.size(); i++){
        if(detectResultVector[i].classId == 0){
            planeResultClassVector.push_back(detectResultVector[i]);
        }else if(detectResultVector[i].classId == 1){
            tankResultClassVector.push_back(detectResultVector[i]);
        }else if(detectResultVector[i].classId == 2){
            shipResultClassVector.push_back(detectResultVector[i]);
        }
    }

    ///删除重复计算的目标
    getRepeatDetect(planeResultClassVector);
    getRepeatDetect(tankResultClassVector);
    getRepeatDetect(shipResultClassVector);

    string planeTotalNumString = "Plane: "+ to_string(planeResultClassVector.size());
    string tankTotalNumString = "Tank: "+ to_string(tankResultClassVector.size());
    string shipTotalNumString = "Ship: "+ to_string(shipResultClassVector.size());

    ///在大图上绘制文字
    putText(yoloResultMat, planeTotalNumString + tankTotalNumString + shipTotalNumString, Point(20, 20),CV_FONT_HERSHEY_COMPLEX,1.0,0,2);


    /***
    * Part8 在大图中绘制bounding box
    */
    /// 绘制bounding box
    for(int i = 0; i < detectResultVector.size(); i++){

        drawPred(detectResultVector[i].classId, detectResultVector[i].center_distance, detectResultVector[i].left, detectResultVector[i].top
                , (detectResultVector[i].left + detectResultVector[i].width), (detectResultVector[i].top + detectResultVector[i].height), yoloResultMat);
    }

    cout << ">>>> Bounding box Print success" << endl;

    ///写最终结果大图到本地
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

/**
 * 删除重复的目标，判断条件：左上角横纵坐标相差小于20
 * @param detectVectorClass
 */
void getRepeatDetect(vector<detect_result_class> & detectVectorClass){
    for(int i = 0; i < detectVectorClass.size(); i++){
        for(int j = i + 1; j < detectVectorClass.size(); j++){
            if(fabs(detectVectorClass[i].left - detectVectorClass[j].left) < 20 && fabs(detectVectorClass[i].top - detectVectorClass[j].top) < 20){
                detectVectorClass.erase(detectVectorClass.begin() + j);
            }
        }
    }
}
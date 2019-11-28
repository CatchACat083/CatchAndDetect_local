
#include <iostream>

#include <darknet.h>
#include "darknetProcess.h"
#include "imageProcess.h"
#include "imageCut.h"
#include "time.h"
#include "circleProcess.h"
#include "amimonLocalImg.h"
#include "insightCatchImg.h"

#define TEST_IMG_PATH "/home/ubuntu/git/CatchAndDetect/img.txt"         //local步骤中image list的存放位置
#define YOLO_CFG_PATH "/home/ubuntu/git/darknet-master/cfg/yolov3test1-voc.cfg"
#define YOLO_WEG_PATH "/home/ubuntu/git/darknet-master/backup/yolov3tuchuan-voc_10000.weights"
#define YOLO_NAM_PATH "/home/ubuntu/git/darknet-master/data/coco.names"
#define IMG_CAL_NAME "/home/ubuntu/git/CatchAndDetect/calibration/"     //采集并完成畸变校正的图像位置
#define IMG_STI_NAME "/home/ubuntu/git/CatchAndDetect/stitcher/"        //拼接完成的大图的位置
#define IMG_YOLO_NAME "/home/ubuntu/git/CatchAndDetect/yoloresult/"     //yolo最终得到结果的大图的位置
#define RTSP_ADD "rtsp://192.168.2.220:554/stream/1"                    //insight图传的rtsp地址
#define QGC_TXT_PATH "/home/ubuntu/Project/2019Match_ZSKT/Result/QGCtxt/"   //QGC保存的txt文件的地址

#define IMG_NUM_TOTAL 6     //规划图像采集的数量



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
     * Part1 读取图像 根据预先设定的图像畸变参数对connex数传采集的图像进行预处理
     * Q: 图像预处理部分只做了畸变处理并去除connex黑边 可以根据需要再添加
     */
    string imgListPath = TEST_IMG_PATH;
    string imgFileFolder = IMG_CAL_NAME;
    imageLocal_amiomon(imageCatchesVector,imgListPath,imgFileFolder);

    /****
     * insight图传qgc采集
    string rtspAddress = RTSP_ADD;
    string qgcTxtFolder = QGC_TXT_PATH;
    string imgFileFolder = IMG_CAL_NAME;
    int imgTotalNum = IMG_NUM_TOTAL;
    imageCatch_insight(imageCatchesVector,rtspAddress,qgcTxtFolder,imgFileFolder,imgTotalNum);
     **/


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

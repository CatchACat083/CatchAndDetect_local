//
// Created by Helu on 6/12/2019.
// 由于愚蠢的规则导向，以及不智慧的stitcher类无法完成比赛任务，因此设计强行拼接类。
// 强行拼接类在预先得知一行图像之间重叠像素和两行图像重叠的像素数（可以通过航线间距、图像采集点、相机在固定高度下的采集像素大小得到），
// 获得最终的“拼接”大图
//

#include "imageForceStitch.h"

#define SingleYoloFolder "/home/ubuntu/Project/2019Match_ZSKT/Result/singleyoloresult/" //强行拼接的图的结果位置
#define WidthRepeat 290     //一行图像之间重叠的像素数
#define HeightRepeat 350    //两行图像之间重叠的像素数

string getTimeString_IFS();

void imageForceStitch()
{
    vector <Mat> yoloForceStitchVector;     //待拼接的图像vector，全部图像已经经过yolo检测并标志好boundingbox
    Mat yoloImgMat;                  //待拼接的单张mat
    int thisImgNum = 0;                     //当前采集的图像数量
    struct stat buffer;                     //文件stat buffer
    /**
     *读取yolo的20张图放入stitch中
     */
    for (; thisImgNum < 20; thisImgNum ++) {
            string InImgName = SingleYoloFolder + to_string(thisImgNum) + ".jpg";
            ///如果查询文件夹发现jpg文件
            if (stat(InImgName.c_str(), &buffer) == 0) {
                yoloImgMat = imread(InImgName);
                yoloForceStitchVector.push_back(yoloImgMat);
            }
        }

    Mat yoloForceStitchMat; //强行拼接的结果Mat
    yoloForceStitchMat.create(1190 * 2 - HeightRepeat, 614 * 10 - WidthRepeat*9,  yoloImgMat.type());

    ///拼接第一行
    for(int i = 0; i < 10; i ++)
    {
        Mat r1 = yoloForceStitchMat(Rect(i* (614 - WidthRepeat), 0, 614, 1190));
        yoloForceStitchVector[i].copyTo(r1);
        cout << ">>>> Force stitch image" << i << endl;
    }

    ///拼接第二行
    for(int j = 10; j < 20; j ++)
    {
        Mat r2 = yoloForceStitchMat(Rect((19 - j) * (614 - WidthRepeat), 1190 - HeightRepeat, 614, 1190));
        transpose(yoloForceStitchVector[j], yoloForceStitchVector[j]);
        flip(yoloForceStitchVector[j], yoloForceStitchVector[j], 1);
        transpose(yoloForceStitchVector[j], yoloForceStitchVector[j]);
        flip(yoloForceStitchVector[j], yoloForceStitchVector[j], 1);
        yoloForceStitchVector[j].copyTo(r2);
        cout  << ">>>> Force stitch image" << j << endl;
    }

    string name = "stitch" + getTimeString_IFS() + ".jpg";
    string OutImgName = SingleYoloFolder + name;
    imwrite(OutImgName, yoloForceStitchMat);
}

string getTimeString_IFS(){
    struct tm *myTimeStruct;
    time_t myTime;
    myTime = time(NULL);
    myTimeStruct = localtime(&myTime);

    //string timeString = NULL;
    string timeString = to_string(myTimeStruct->tm_year + 1900) + to_string(myTimeStruct->tm_mon + 1) + to_string(myTimeStruct->tm_mday) + to_string(myTimeStruct->tm_hour)
                        + to_string(myTimeStruct->tm_min) + to_string(myTimeStruct->tm_sec);
    return timeString;
}

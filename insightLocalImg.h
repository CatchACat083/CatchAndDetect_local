//
// Created by ubuntu on 28/11/2019.
//
// Created by Bosen on 28/11/2019.
//
#ifndef CATCHANDDETECT_INSIGHTLOCALIMG_H
#define CATCHANDDETECT_INSIGHTLOCALIMG_H
#include <iostream>

#include <darknet.h>
#include "darknetProcess.h"
#include "imageProcess.h"
#include "imageCut.h"
#include "time.h"
#include "circleProcess.h"
#include <sys/stat.h>


#define INSIGHT_IN_HEI 720 //CONNEX图传传入的图像_高度
#define INSIGHT_IN_WID 1280 //CONNEX图传传入的图像_宽度
#define INSIGHT_CAL_HEI 614 //CONNEX图传校正后的图像_高度
#define INSIGHT_CAL_WID 1190 //CONNEX图传校正后的图像_宽度


//去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
#define INSIGHT_CAL_REZ_TOP 53
#define INSIGHT_CAL_REZ_LEFT 45
#define INSIGHT_CAL_REZ_HEI 614
#define INSIGHT_CAL_REZ_WID 1190

using namespace std;
using namespace cv;

void imageLocal_insight(vector<Mat> & imageCatchesVector,string imgListPath,string imgFileFloder);

string getTimeString_ILI();


#endif //CATCHANDDETECT_INSIGHTLOCALIMG_H

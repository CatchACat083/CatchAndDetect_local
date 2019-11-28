//
// Created by ubuntu on 28/11/2019.
//

#include <iostream>

#include <darknet.h>
#include "darknetProcess.h"
#include "imageProcess.h"
#include "imageCut.h"
#include "time.h"
#include "circleProcess.h"
#include <sys/stat.h>

using namespace std;
using namespace cv;

#ifndef CATCHANDDETECT_INSIGHTCATCHIMG_H
#define CATCHANDDETECT_INSIGHTCATCHIMG_H

#define INSIGHT_IN_HEI 1080 //CONNEX图传传入的图像_高度
#define INSIGHT_IN_WID 1920 //CONNEX图传传入的图像_宽度
#define INSIGHT_CAL_HEI 980 //CONNEX图传校正后的图像_高度
#define INSIGHT_CAL_WID 1870 //CONNEX图传校正后的图像_宽度


//去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
#define INSIGHT_CAL_REZ_TOP 50
#define INSIGHT_CAL_REZ_LEFT 25
#define INSIGHT_CAL_REZ_HEI 980
#define INSIGHT_CAL_REZ_WID 1870




void imageCatch_insight(vector<Mat> imageCatchesVector, string rtspAddress, string qgcTxtFolder, string imgFileFolder, int imgTotalNum);

string getTimeString();

void imageReMap(Mat srcMat, Mat distMat);


#endif //CATCHANDDETECT_INSIGHTCATCHIMG_H

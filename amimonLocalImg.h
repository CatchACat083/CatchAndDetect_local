//
// Created by ubuntu on 28/11/2019.
//

#ifndef CATCHANDDETECT_AMIMONLOCALIMG_H
#define CATCHANDDETECT_AMIMONLOCALIMG_H


#include <iostream>

#include <darknet.h>
#include "darknetProcess.h"
#include "imageProcess.h"
#include "imageCut.h"
#include "time.h"
#include "circleProcess.h"
#include <sys/stat.h>

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

using namespace std;
using namespace cv;

void imageLocal_amiomon(vector<Mat> imageCatchesVector,string imgListPath,string imgFileFloder);

string getTimeString();


#endif //CATCHANDDETECT_AMIMONLOCALIMG_H

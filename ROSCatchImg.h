//
// Created by ubuntu on 2/12/2019.
//

#ifndef CATCHANDDETECT_LOCAL_ROSCATCHIMG_H
#define CATCHANDDETECT_LOCAL_ROSCATCHIMG_H


#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "time.h"
#include <sys/stat.h>
#include "imageProcess.h"

using namespace std;
using namespace cv;
#define INSIGHT_IN_HEI 720 //CONNEX图传传入的图像_高度
#define INSIGHT_IN_WID 1280 //CONNEX图传传入的图像_宽度
#define INSIGHT_CAL_HEI 614 //CONNEX图传校正后的图像_高度
#define INSIGHT_CAL_WID 1190 //CONNEX图传校正后的图像_宽度


//去除畸变造成的黑边_黑边在高度和宽度上都出现_高度上可用像素50_1030_宽度上可用像素25_1895
#define INSIGHT_CAL_REZ_TOP 53
#define INSIGHT_CAL_REZ_LEFT 45
#define INSIGHT_CAL_REZ_HEI 614
#define INSIGHT_CAL_REZ_WID 1190




void imageCatch_ROSinsight(vector<Mat> & imageCatchesVector, string imgInFolder, string imgOutFolder, int imgTotalNum);

string getTimeString_RCI();

void imageReMap(Mat & srcMat, Mat & distMat) ;

#endif //CATCHANDDETECT_LOCAL_ROSCATCHIMG_H

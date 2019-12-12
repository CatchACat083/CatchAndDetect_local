//
// Created by Bosen on 4/12/2019.
//

#ifndef CATCHANDDETECT_LOCAL_IMAGEPREPROCESS_H
#define CATCHANDDETECT_LOCAL_IMAGEPREPROCESS_H

#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;


void imageEqual2(vector<Mat> & imageVectorSrc, vector<Mat> & imageVectorDis);
void contrastStretch(Mat &srcImage, Mat &dstImage);


#endif //CATCHANDDETECT_LOCAL_IMAGEPREPROCESS_H

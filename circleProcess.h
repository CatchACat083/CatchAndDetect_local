//
// Created by Bosen on 2019/10/29.
//

#ifndef CATCHANDDETECT_CIRCLEPROCESS_H
#define CATCHANDDETECT_CIRCLEPROCESS_H

#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;

struct circle_class{
    float left; //圆心x坐标(在整个大图中的坐标)
    float top; //圆心y坐标(在整个大图中的坐标)
    float radius; // 半径(以像素点表示)
};

circle_class detectCircle(Mat & srcImage,vector<circle_class> & circleClass, Point2f pointInSitich, int i);



#endif //CATCHANDDETECT_CIRCLEPROCESS_H

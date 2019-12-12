//
// Created by Bosen on 2019/10/22.
//

#ifndef CATCHANDDETECT_IMAGEPROCESS_H
#define CATCHANDDETECT_IMAGEPROCESS_H

#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace std;
using namespace cv;

void resizeImage(Mat & img_src, Mat & img_rez, int row, int col, int height, int width);
//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
Point2f getTransformPoint(const Point2f point,const Mat & homo);
//计算变换后右图的四个角点坐标值
void calculateCorners(const Mat & homo, const Mat & image, Point2f corners[]);
//图像拼接
void imageStitch(Mat & imagesrc, Mat & imagetes);

struct stitch_class{
    int col_num; //
    int row_num;
    int left; //左上角点x坐标
    int top; //左上角点y坐标
    
};


#endif //CATCHANDDETECT_IMAGEPROCESS_H

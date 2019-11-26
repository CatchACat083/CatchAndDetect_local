//
// Created by lin083 on 2019/10/25.
//

#ifndef CATCHANDDETECT_IMAGECUT_H
#define CATCHANDDETECT_IMAGECUT_H

#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;


/**
 * 定义结构yolo_image为进入yolo进行目标识别的图像
 */
struct yolo_image{
    Mat yolo_image; //图像Mat
    int width; //图像宽度
    int height; //图像高度
    Point2f point_in_sitich; //在拼接原图上左上点的坐标
};



/**
 * 图像分割，将stitch后的大图像分为若干张小图传入yolo中
 * @param stitch_image //图像拼接后的大图像
 * @param yolo_images //裁切后的图像数组以vector显示
 */
void imageCut(Mat & stitch_image, vector<yolo_image> & yolo_images);

#endif //CATCHANDDETECT_IMAGECUT_H

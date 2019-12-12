//
// Created by Bosen on 2019/10/15.
//

#ifndef CATCHANDDETECT_DARKNETPROCESS_H
#define CATCHANDDETECT_DARKNETPROCESS_H

#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;


struct detect_class{
    int classId; //class id 类序号
    string label; //class label 类标识
    float confidences; //confidences 概率
    int left; //左上角点x坐标_在小图中
    int top; //左上角点y坐标_在小图中
    int width; //宽度
    int height; //高度
};



void postprocess(Mat& frame, const vector<Mat>& outs, vector<detect_class>& detectClassList);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

vector<String> getOutputsNames(const Net& net);

void detect_image(Mat& frame, Mat& dst_frame, vector<detect_class>& detectClassList, string modelWeights, string modelConfiguration, string classesFile);



#endif //CATCHANDDETECT_DARKNETPROCESS_H

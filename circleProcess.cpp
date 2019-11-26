//
// Created by lin083 on 2019/10/29.
//
# include "circleProcess.h"

circle_class detectCircle(Mat & srcImage, vector<circle_class> & circleClass, Point2f pointInSitich){
    circle_class thisCircle; //本图像中的一个红色圆

    //降低图像噪声
    GaussianBlur(srcImage, srcImage, Size(5,5),2,2);

    vector<Vec3f> circlesHough;

    Mat srcImageHSV;
    //将图像变为HSV
    cvtColor(srcImage, srcImageHSV, CV_BGR2HSV);

    /***
     * 鉴于red的hsv部分分为两部分，因此使用两个mask分别处理两部分，最后使用按位”或“运算组合
     */
    //lower mask
    Mat lowerMask;
    inRange(srcImageHSV, Scalar(0,43,46), Scalar(20,255,240),lowerMask);//一般选择h值和s值，并设定v值为20-255
    //upper mask
    Mat upperMask;
    //一般选择h值和s值，并设定v值为20-255
    inRange(srcImageHSV, Scalar(150,43,46), Scalar(180,255,240),upperMask);

    //lower mask dilate
    Mat lowerDilate;
    vector<Vec3f> lowerCirclesHough;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    dilate(lowerMask, lowerDilate, element, Point(-1,-1),
           3, BORDER_CONSTANT);
    /*
     * 膨胀运算
     * element = 自定义核 shape = MORPH_ELLIPSE 椭圆形
     * iterations = 3 膨胀的次数
     * borderType = BORDER_CONSTANT
     */
    //HoughCircles(lowerDilate, lowerCirclesHough, CV_HOUGH_GRADIENT, 1, 100, 100, 30, 0, 0);
    /*
     * hough运算寻找圆
     * dp =1时，累加器和输入图像具有相同的分辨率。如果=2，累加器便有输入图像一半那么大的宽度和高度
     * minDist 为霍夫变换检测到的圆的圆心之间的最小距离，即让我们的算法能明显区分的两个不同圆之间的最小距离
     * param1 默认值100。它是method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半
     * param2，也有默认值100。它是method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法，它表示在检测阶段圆心的累加器阈值。
     *           它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。
     * minRadius，默认值0，表示圆半径的最小值。
     * maxRadius，也有默认值0，表示圆半径的最大值。*/

    //upper mask dilate
    Mat upperDilate;
    vector<Vec3f>  upperCirclesHough;
    dilate(upperMask, upperDilate, element, Point(-1,-1),
           3, BORDER_CONSTANT);
    //HoughCircles(upperDilate, upperCirclesHough, CV_HOUGH_GRADIENT, 1, 100, 100, 30, 0, 0);

    //combile upper mask and lower mask
    Mat fullDilate;
    bitwise_or(lowerDilate,upperDilate,fullDilate);
    vector<Vec3f>  fullCirclesHough;

    ///hough circle in full dildate mat
    ///此处需要根据实际值更改maxradius 和 maxradius
    HoughCircles(fullDilate, fullCirclesHough, CV_HOUGH_GRADIENT, 1, 500, 75, 9, 0, 0);

    //将该张图片中检测的圆加入到circleClass中
    for(int i = 0; i< fullCirclesHough.size(); i++) {
        circle_class thiscircleClass;
        thiscircleClass.left = fullCirclesHough[i][0] + pointInSitich.x;
        thiscircleClass.top = fullCirclesHough[i][1] + pointInSitich.y;
        thiscircleClass.radius = fullCirclesHough[i][2];
        circleClass.push_back(thiscircleClass);
    }

}
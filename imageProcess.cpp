//
// Created by Bosen on 2019/10/22.
// 定义了一些基础的图像处理函数，比如裁切image、手动imagestitch等
//

#include "imageProcess.h"


/*
 * 由于connex数传输出图像宽度和原图不一致重新设置
 * row 裁剪起点y坐标, col 裁剪起点x坐标, height 裁剪高度, width 裁剪宽度
 * return img_dst.size(1920, 1080)
 */
void resizeImage(Mat & img_src, Mat & img_rez, int row, int col, int height, int width)
{
    Rect area(col, row, width, height);
    Mat img_region = img_src(area);
    resize(img_region, img_rez, img_rez.size(), 0,0,INTER_LINEAR);
    return;
}

//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
//point 原始图像上的点, homo 变换矩阵
Point2f getTransformPoint(const Point2f point,const Mat & homo)
{
    Mat originelP, targetP;
    originelP = (Mat_<double>(3,1) << point.x, point.y, 1.0);
    targetP = homo * originelP;
    float x = targetP.at<double>(0,0) / targetP.at<double>(2,0);
    float y = targetP.at<double>(1,0) / targetP.at<double>(2,0);
    return Point2f(x,y);
}

//定位右侧图像变换之后的四个顶点位置
//image 图像, homo 变换矩阵
void calculateCorners(const Mat & homo, const Mat & image, Point2f corners[])
{
    //四个顶点,corners[0] - corners[3] 分别为left_top,right_top,left_bottom,right_bottom

    //left_top
    Point2f left_top = Point(0,0);
    corners[0] = getTransformPoint(left_top, homo);
    //cout << corners[0].x <<","<< corners[0].y << endl;

    //right_top
    Point2f right_top = Point(image.cols, 0);
    corners[1] = getTransformPoint(right_top, homo);
    //cout << corners[1].x <<","<< corners[1].y << endl;

    //right_bottom
    Point2f right_bottom = Point(image.cols, image.rows);
    corners[2] = getTransformPoint(right_bottom, homo);
    //cout << corners[2].x <<","<< corners[2].y << endl;

    //left_bottom
    Point2f left_bottom = Point(0, image.rows);
    corners[3] = getTransformPoint(left_bottom, homo);
    //cout << corners[3].x <<","<< corners[3].y << endl;

    return;
}

void imageStitch(Mat & imagesrc, Mat & imagetes)
{
    Mat image1 = imagesrc.clone();
    Mat image2 = imagetes.clone();

    vector<Point2f> image1Points;
    vector<Point2f> image2Points;

    //提取特征点
    Ptr<Feature2D> SurfDetector = xfeatures2d::SURF::create(800);
    //cv::Ptr<Feature2D> f2d = xfeatures2d::SURF::create();
    //cv::Ptr<Feature2D> f2d = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    SurfDetector -> detect(image1, keypoints1);
    SurfDetector -> detect(image2, keypoints2);

    //计算特征点描述符
    Mat descriptors1, descriptors2;
    SurfDetector -> compute(image1, keypoints1, descriptors1);
    SurfDetector -> compute(image2, keypoints2, descriptors2);

    //获取匹配特征点，提取最优配对
    FlannBasedMatcher matcher;
    //BFMatcher matcher;
    vector<DMatch> matchPoints;
    matcher.match(descriptors1,descriptors2,matchPoints,Mat());
    //获取最优匹配特征点:匹配点距离<3*最小匹配距离
    vector<Point2f> imageMatcher1,imageMatcher2;
    cout << ">>>> get surf matcher success" << endl;

    //获取最大/最小匹配距离
    double maxDistance = matchPoints[0].distance;
    double minDistance = matchPoints[0].distance;
    for(int i = 0; i < matchPoints.size(); i++)
    {
        double dist = matchPoints[i].distance;
        if( dist < minDistance ) minDistance = dist;
        if( dist > maxDistance ) maxDistance = dist;
    }

    int invertNum = 0;  //统计image2.x > image1.x 的匹配点对的个数，来判断image1是否在右侧

    //从最有匹配中获得匹配点
    for(int i = 0; i < matchPoints.size(); i++)
    {
        Point2f pt1 = keypoints1[matchPoints[i].queryIdx].pt;
        Point2f pt2 = keypoints2[matchPoints[i].trainIdx].pt;

        if(matchPoints[i].distance < 3 * minDistance) {
            imageMatcher1.push_back(pt1);
            imageMatcher2.push_back(pt2);
            if(pt2.x > pt1.x) invertNum++;//统计匹配点的左右位置关系，来判断图1和图2的左右位置关系
        }
    }

    //获取图像1到图像2的投影映射矩阵，尺寸为3*3
    Mat homo = findHomography(imageMatcher2,imageMatcher1,CV_RANSAC);
    //Mat adjustMat = (Mat_<double>(3,3)<<1.0,0,image2.cols,0,1.0,0,0,0,1.0);
    //Mat adjustHomo = adjustMat * homo;

    /*程序中计算出的变换矩阵homo用来将image2中的点变换为image1中的点，正常情况下image1应该是左图，image2应该是右图。
      此时image2中的点pt2和image1中的对应点pt1的x坐标的关系基本都是：pt2.x < pt1.x
      若用户打开的image1是右图，image2是左图，则image2中的点pt2和image1中的对应点pt1的x坐标的关系基本都是：pt2.x > pt1.x
      所以通过统计对应点变换前后x坐标大小关系，可以知道image1是不是右图。
      如果image1是右图，将image中的匹配点经H的逆阵H_IVT变换后可得到image2中的匹配点*/

    Mat imageTransform;

    //定义左图,左图点,右图,右图点,均为对原有Mat的引用
    if(! homo.empty()){
        cout << ">>>> create homography success" << endl;
        Mat imageLeft;
        Mat imageRight;
        vector<Point2f> imagePointsLeft;
        vector<Point2f> imagePointsRight;

        //若pt2.x > pt1.x的点的个数大于内点个数的80%，则认定image1中是右图
        if(invertNum > imageMatcher1.size() * 0.8){
            Mat homoIVT;
            invert(homo, homoIVT);//变换矩阵的逆矩阵
            homo = homoIVT;

            imageLeft = image2;
            imageRight = image1;
            imagePointsLeft = image2Points;
            imagePointsRight = image1Points;

        }
        else{
            imageLeft = image1;
            imageRight = image2;
            imagePointsLeft = image1Points;
            imagePointsRight = image2Points;
        }

        //获得变换后右图的四角点在新图像上的位置
        Point2f corners[4] = {Point(0,0), Point(0,0), Point(0,0), Point(0,0)};
        calculateCorners(homo, imageRight, corners);

        //计算新图像中各点的位置
        for(int i = 0; i < imagePointsRight.size(); i++){
            imagePointsLeft.push_back(getTransformPoint(imagePointsRight[i],homo));
        }

        //变换后图像的宽度
        int warpWidth = MAX(corners[1].x, corners[2].x);
        if(warpWidth < 1920){
            cout << "!!!! warp image failure" << endl;
            return;
        }else{
            cout << ">>>> warp image successful" << endl;
        }

        //拼接右侧图像
        warpPerspective(imageRight, imageTransform, homo, Size(warpWidth, imageLeft.rows));
        //拼接左侧图像
        imageLeft.copyTo(Mat(imageTransform,Rect(0,0,imageLeft.cols,imageLeft.rows)));

        /////imageTransform 拼接完成的图像
        //namedWindow("拼接结果",0);
        //imshow("拼接结果",imageTransform);
        imagesrc = imageTransform.clone();
        cout << ">>>> stitch image successful" << endl;
    }
    else{
        cout << "!!!! create homography failure" << endl;
        cout << "!!!! stitch image failure" << endl;
    }
    return;
}

//void imageStitchDivide(vector<Mat> &image_stitchers, vector<)
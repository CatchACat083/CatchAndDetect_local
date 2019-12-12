//
// Created by Bosen on 4/12/2019.
// 因为比赛场地的图像可供拼接的特征过少，因此考虑通过图像增强（预处理）增加图像特征进行拼接
// 此处实现了若干图像增强算法
//

#include "imagePreProcess.h"

/**
 * 图像预处理：拉普拉斯算子变换
 */
void imageLpulasi(vector<Mat> & imageVectorSrc, vector<Mat> & imageVectorDis){
    for(int i = 0; i < imageVectorSrc.size(); i ++){
        Mat imgSrc = imageVectorSrc[i].clone(); //原图像
        Mat imgDist;                            //目标图像

        ///进行图像拉普拉斯算子变换
        Mat kernel =  (Mat_<float>(3,3)<<0,-1,0,0,4,0,0,-1,0);
        filter2D(imgSrc, imgDist, CV_8UC3, kernel);

        //string imageOutName = "/home/ubuntu/Project/2019Match_ZSKT/Result/ManualCatch/preprocess"+ to_string(i) + ".jpg";
        //imwrite(imageOutName, imgDist);

        imageVectorDis.push_back(imgDist);
    }
    cout << ">>>>> image Lpulasi perprocess success" << endl;
}

/***
 * 图像预处理：gama变换
 * @param imageVectorSrc
 * @param imageVectorDis
 */
void imageGama(vector<Mat> & imageVectorSrc, vector<Mat> & imageVectorDis) {
    for (int i = 0; i < imageVectorSrc.size(); i++) {
        Mat imgSrc = imageVectorSrc[i].clone(); //原图像
        Mat imgDist;                            //目标图像

        for (int i = 0; i < imgSrc.rows; i++) {
            for (int j = 0; j < imgSrc.cols; j++) {
                imgDist.at<Vec3f>(i, j)[0] =
                        (imgSrc.at<Vec3b>(i, j)[0]) * (imgSrc.at<Vec3b>(i, j)[0]) * (imgSrc.at<Vec3b>(i, j)[0]);
                imgDist.at<Vec3f>(i, j)[1] =
                        (imgSrc.at<Vec3b>(i, j)[1]) * (imgSrc.at<Vec3b>(i, j)[1]) * (imgSrc.at<Vec3b>(i, j)[1]);
                imgDist.at<Vec3f>(i, j)[2] =
                        (imgSrc.at<Vec3b>(i, j)[2]) * (imgSrc.at<Vec3b>(i, j)[2]) * (imgSrc.at<Vec3b>(i, j)[2]);
            }
        }
        normalize(imgDist, imgDist, 0, 255, CV_MINMAX);
        convertScaleAbs(imgDist, imgDist);

        imageVectorDis.push_back(imgDist);
    }

    cout << ">>>>> image Gama perprocess success" << endl;
}

/***
 * 图像预处理，gama预处理
 * @param imageVectorSrc
 * @param imageVectorDis
 */
void imageGama2(vector<Mat> & imageVectorSrc, vector<Mat> & imageVectorDis) {
    for (int i = 0; i < imageVectorSrc.size(); i++) {
        Mat imgSrc = imageVectorSrc[i].clone(); //原图像
        Mat imgDist;                            //目标图像

        Mat imgSrcB, imgSrcG, imgSrcR;
        vector<Mat> channels;
        split(imgSrc, channels);


        int c = 300;
        double r = 0.10;

        for (int i = 0; i < channels.size(); i++) {
            channels[i].convertTo(channels[i], CV_32F, 1.0/255.0);
            pow(channels[i] / 255, r, channels[i]);
            channels[i] = channels[i] * c;
        }
        merge(channels, imgDist);

        imageVectorDis.push_back(imgDist);
    }
    cout << ">>>>> image Gama perprocess success" << endl;
}

/***
 * 图像预处理：直方图均衡化，对YCRCB中的Y通道进行变换
 * @param imageVectorSrc
 * @param imageVectorDis
 */
void imageEqual(vector<Mat> & imageVectorSrc, vector<Mat> & imageVectorDis){
    for (int i = 0; i < imageVectorSrc.size(); i++) {
        Mat imgSrc = imageVectorSrc[i].clone(); //原图像
        Mat imgDist;

        Mat imgYCrCb;

        cvtColor(imgSrc, imgYCrCb, COLOR_BGR2YCrCb);
        std::vector<Mat> channels;
        split(imgYCrCb, channels);

        equalizeHist(channels[0],channels[0]);
        merge(channels, imgDist);
        cvtColor(imgDist,imgDist,COLOR_YCrCb2BGR);

        imageVectorDis.push_back(imgDist);
    }
    cout << ">>>>> image YCRCB Equal perprocess success" << endl;
}

/***
 * 图像预处理：直方图均衡化，对HSV中的V通道进行变换
 * @param imageVectorSrc
 * @param imageVectorDis
 */
void imageEqual2(vector<Mat> & imageVectorSrc, vector<Mat> & imageVectorDis){
    for (int i = 0; i < imageVectorSrc.size(); i++) {
        Mat imgSrc = imageVectorSrc[i].clone(); //原图像
        Mat imgDist;

        Mat imgHSV;

        cvtColor(imgSrc, imgHSV, COLOR_BGR2HSV);
        std::vector<Mat> channels;
        split(imgHSV, channels);

        //equalizeHist(channels[0],channels[0]);
        //equalizeHist(channels[1],channels[1]);
        equalizeHist(channels[2],channels[2]);
        //contrastStretch2(channels[2],channels[2]);
        //equalizeHist(channels[2],channels[2]);
        merge(channels, imgDist);
        cvtColor(imgDist,imgDist,COLOR_HSV2BGR);

        //string imageOutName = "/home/ubuntu/Project/2019Match_ZSKT/Result/1204/process/"+ to_string(i) + ".jpg";
        //cout << "图像增强   "  << i << endl;
        //imwrite(imageOutName, imgDist);

        imageVectorDis.push_back(imgDist);
    }
    cout << ">>>>> image HSV Equal perprocess success" << endl;
}

/***
 * 图像预处理：图像线性变换
 * @param srcImage
 * @param dstImage
 */
void imageContrastStretch(Mat &srcImage, Mat &dstImage)
{
    ///计算图像的最大最小值
    double pixMin,pixMax;
    cv::minMaxLoc(srcImage,&pixMin,&pixMax);

    ///生成LUT表
    cv::Mat lut(1, 256, CV_8U);
    for(int i = 0; i < 256; i++ ){
        if (i < pixMin) lut.at<uchar>(i)= 0;
        else if (i > pixMax) lut.at<uchar>(i)= 255;
        else lut.at<uchar>(i)= static_cast<uchar>(255.0 * 0.9 * (i-pixMin)/(pixMax-pixMin) - 2);
        //else lut.at<uchar>(i)= static_cast<uchar>(255.0 * 0.5 * i - 2);
    }
    ///对图像应用Lat表
    LUT( srcImage, lut, dstImage );
    cout << ">>>>> image Contrast Stretch perprocess success" << endl;
}

/*int main(){

    string imgListPath = "/home/ubuntu/Project/2019Match_ZSKT/CatchAndDetect_local/img.txt";
    //图像输入
    ifstream fileInputstream(imgListPath);
    string imgNameString;
    vector<Mat> imageVectorSrc;
    vector<Mat> imageVectorDist;
    Mat imgInputMat;

    while(getline(fileInputstream, imgNameString)) {
        imgInputMat = imread(imgNameString);
        imageVectorSrc.push_back(imgInputMat);
    }

    cout << imageVectorSrc.size() << endl;
    imageEqual2(imageVectorSrc,imageVectorDist);

    for(int i = 0; i < imageVectorSrc.size(); i++){
        imshow("windows", imageVectorDist[i]);
        string imageOutName = "/home/ubuntu/Project/2019Match_ZSKT/Result/1204/process/"+ to_string(i) + ".jpg";
        imwrite(imageOutName, imageVectorDist[i]);
    }

}*/
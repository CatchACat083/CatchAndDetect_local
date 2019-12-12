//
// Created by Helu on 2019/10/25.
// 将最后的大图裁切为1190*614大小，输入yolo（为了保证onestage的yolo训练图片和最后的检测图片大小一致）
//
#include "imageCut.h"
#define IMG_CUT_PATH "/home/ubuntu/Project/2019Match_ZSKT/Result/cutresult/"

Rect m_select;

int surewh(Mat stitch_image, int & a, int & b, int & yolowidth, int & yoloheight)//确定分割块数
{
    int totalw = stitch_image.cols;
    int totalh = stitch_image.rows;
    int singlew = 1190;
    int singleh = 614;
    if(totalw%(singlew*2/3)>(singlew/2)){
        a = totalw/(singlew*2/3)+1;}
    else{a = totalw/(singlew*2/3);}
    if(totalh%(singleh*2/3)>(singleh/2)){
        b = totalh/(singleh*2/3)+1;}
    else{b = totalh/(singleh*2/3);}
}

void imageCut(Mat & stitch_image, vector<yolo_image> & yolo_images)
{
    int wcount;//横向裁剪块数
    int hcount;//纵向裁剪块数
    surewh(stitch_image, wcount, hcount, stitch_image.cols, stitch_image.rows);
    cout << wcount <<" "<< hcount << endl;
    Mat proimage;
    yolo_image yoloimage;
    int yolowidth = stitch_image.cols/wcount+1;//分割后每块宽
    int yoloheight = stitch_image.rows/hcount+1;//分割后每块高
    cout << yolowidth <<" "<< yoloheight << endl;
    for(int i=1; i<=wcount*(hcount-1); i++)
    {
        if(i % wcount == 0)//最右侧图像
        {
            m_select = Rect(yolowidth*(wcount-1),yoloheight*(i/wcount-1),stitch_image.cols-yolowidth*(wcount-1),yoloheight*4/3);
            proimage = stitch_image(m_select);
            yoloimage.yolo_image = proimage;
            yoloimage.width = stitch_image.cols-yolowidth*(wcount-1);
            yoloimage.height = yoloheight*4/3;
            yoloimage.point_in_sitich.x = yolowidth*(wcount-1);// 横坐标
            yoloimage.point_in_sitich.y = yoloheight*(i/wcount-1);//纵坐标
            yolo_images.push_back(yoloimage);
            cout << yoloimage.point_in_sitich.x <<" "<< yoloimage.point_in_sitich.y << endl;
        }
        else
        {
            m_select = Rect(yolowidth*(i%wcount-1),yoloheight*(i/wcount),yolowidth*4/3,yoloheight*4/3);//多裁三分之一
            proimage = stitch_image(m_select);
            yoloimage.yolo_image = proimage;
            yoloimage.width = yolowidth*4/3;
            yoloimage.height = yoloheight*4/3;
            yoloimage.point_in_sitich.x = yolowidth*(i%wcount-1);
            yoloimage.point_in_sitich.y = yoloheight*(i/wcount);
            yolo_images.push_back(yoloimage);
            cout << yoloimage.point_in_sitich.x <<" "<< yoloimage.point_in_sitich.y << endl;
        }
    }
    for(int j=wcount*(hcount-1)+1; j<=wcount*hcount; j++)//最下侧图像
    {
        if(j % wcount == 0)//最右下侧图像
        {
            m_select = Rect(yolowidth*(wcount-1),yoloheight*(hcount-1),stitch_image.cols-yolowidth*(wcount-1),stitch_image.rows-yoloheight*(hcount-1));
            proimage = stitch_image(m_select);
            yoloimage.yolo_image = proimage;
            yoloimage.width = stitch_image.cols-yolowidth*(wcount-1);
            yoloimage.height = stitch_image.rows-yoloheight*(hcount-1);
            yoloimage.point_in_sitich.x = yolowidth*(wcount-1);
            yoloimage.point_in_sitich.y = yoloheight*(hcount-1);
            yolo_images.push_back(yoloimage);
            //cout << yoloimage.point_in_sitich.x <<" "<< yoloimage.point_in_sitich.y << endl;
        }
        else
        {
            m_select = Rect(yolowidth*(j%wcount-1),yoloheight*(j/wcount),yolowidth*4/3,stitch_image.rows-yoloheight*(hcount-1));
            proimage = stitch_image(m_select);
            yoloimage.yolo_image = proimage;
            yoloimage.width = yolowidth*4/3;
            yoloimage.height = stitch_image.rows-yoloheight*(hcount-1);
            yoloimage.point_in_sitich.x = yolowidth*(j%wcount-1);
            yoloimage.point_in_sitich.y = yoloheight*(j/wcount);
            yolo_images.push_back(yoloimage);
            //cout << yoloimage.point_in_sitich.x <<" "<< yoloimage.point_in_sitich.y << endl;
        }
    }
    for(int k=0; k<wcount*hcount; k++)
    {
        string imgName = IMG_CUT_PATH + to_string(k) + ".jpg";
        imwrite(imgName,yolo_images[k].yolo_image);
    }
}
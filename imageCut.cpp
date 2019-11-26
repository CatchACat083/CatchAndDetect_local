//
// Created by lin083 on 2019/10/25.
//
#include "imageCut.h"

Rect m_select;


void imageCut(Mat & stitch_image, vector<yolo_image> & yolo_images)
{
    int wcount = 3;//横向裁剪块数
    int hcount = 6;//纵向裁剪块数
    Mat proimage;
    yolo_image yoloimage;
    int yolowidth = stitch_image.cols/wcount+1;//分割后每块宽
    int yoloheight = stitch_image.rows/hcount+1;//分割后每块高
    //cout << yolowidth <<" "<< yoloheight << endl;
    for(int i=1; i<=wcount*(hcount-1); i++)
    {
        if(i % wcount == 0)//最右侧图像
        {
            m_select = Rect(yolowidth*(wcount-1),yoloheight*(i/6-1),stitch_image.cols-yolowidth*(wcount-1),yoloheight*4/3);
            proimage = stitch_image(m_select);
            yoloimage.yolo_image = proimage;
            yoloimage.width = stitch_image.cols-yolowidth*(wcount-1);
            yoloimage.height = yoloheight*4/3;
            yoloimage.point_in_sitich.x = yolowidth*(wcount-1);// 横坐标
            yoloimage.point_in_sitich.y = yoloheight*(i/6-1);//纵坐标
            yolo_images.push_back(yoloimage);
            //cout << yoloimage.point_in_sitich.x <<" "<< yoloimage.point_in_sitich.y << endl;
        }
        else
        {
            m_select = Rect(yolowidth*(i%6-1),yoloheight*(i/6),yolowidth*4/3,yoloheight*4/3);//多裁三分之一
            proimage = stitch_image(m_select);
            yoloimage.yolo_image = proimage;
            yoloimage.width = yolowidth*4/3;
            yoloimage.height = yoloheight*4/3;
            yoloimage.point_in_sitich.x = yolowidth*(i%6-1);
            yoloimage.point_in_sitich.y = yoloheight*(i/6);
            yolo_images.push_back(yoloimage);
            //cout << yoloimage.point_in_sitich.x <<" "<< yoloimage.point_in_sitich.y << endl;
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
            m_select = Rect(yolowidth*(j%6-1),yoloheight*(j/6),yolowidth*4/3,stitch_image.rows-yoloheight*(hcount-1));
            proimage = stitch_image(m_select);
            yoloimage.yolo_image = proimage;
            yoloimage.width = yolowidth*4/3;
            yoloimage.height = stitch_image.rows-yoloheight*(hcount-1);
            yoloimage.point_in_sitich.x = yolowidth*(j%6-1);
            yoloimage.point_in_sitich.y = yoloheight*(j/6);
            yolo_images.push_back(yoloimage);
            //cout << yoloimage.point_in_sitich.x <<" "<< yoloimage.point_in_sitich.y << endl;
        }
    }
    for(int k=0; k<wcount*hcount; k++)
    {
        char buff[256];
        sprintf(buff, "/home/ubuntu/git/CatchAndDetect/cutresult/%d.jpg", k);
        puts(buff);
        imwrite(buff,yolo_images[k].yolo_image);
    }
}


//int main(int argc, char** argv)
//{
//    Mat image1 = imread("/home/zhouhelu/left.jpg");
 //   vector<yolo_image> yolo_images;
 //   imageCut(image1, yolo_images);
 //   return 0;
//}

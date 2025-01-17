# Read Me
##  背景与目的
**智胜空天2019比赛科目一**
1. 比赛场景：比赛场地设置起飞区、多目标识别区、降落区三大区域，其中目标识别区大小为50×100米矩形区域，区内随机放置多种三维军事目标仿真模型；降落区大小为2×2米正方形区域，中心以H字母标示，位置不固定。
2. 平台要求：采用小型电动旋翼飞行器，飞行总重量不超过5kg，数量不多于2架；机体材质不限，轴距不超过680mm；不允许使用差分GPS、差分北斗等高精度定位设备；飞行高度不得超过50米。
3. 比赛流程：无人机完成起飞后，必须采用自主控制方式，不能人为操作地面站和遥控器，完成任务后降落在指定区域。图像识别区域内设置中心定位地标，场地内随机布设飞机、车辆、舰船三类军事目标的立体模型，数量若干。比赛成绩按识别准确度、任务完成时间、自主飞行控制等作为评分依据，包括自主控制分、目标侦察分。自主控制评分指标包括飞行准备时间、任务总时间、飞行高度、自主控制、精准降落。目标侦察评分指标包括目标类别、数量、位置、时间。其中位置以目标距中心定位地标的地面距离表示，识别结果在地面站以一张图像显示。
![规则示意图](http://www.sc.sdu.edu.cn/__local/F/F7/56/4C68EF3919DAF7AE8BB399F5906_AC333DE4_6DA68.png)

## 硬件描述
1. 飞机：轴距650mm四旋翼无人机
2. 飞行控制系统：CUAV PX4 V5，运行PX4-v1.8.2固件
3. 相机及图传系统：gopro4相机（采用视频模式窄视场），insight图传
4. 地面站：ubuntu18.04 PC with RTX2080 & i7 9th Gen

## 程序内容
程序主要流程为：
已知：(1)依据比赛场地规划的航路，在航路上平均分布2x10个图像采集点，得到采集点的GPS坐标。(2)飞机飞行高度25米，相机视角始终垂直向下。(3)通过预先标定的方法得到相机的畸变参数

1. 运行ROS节点（ZSKT-S1），其中ROS节点分为两部分，一个是依据已知的采集点的GPS坐标，当飞机抵达采集位置时发送ROS消息，一个利用opencv打开图传的RTSP视频流，当有ROS消息进入时从视频流中截图并存放在本机

2. 运行检测识别算法程序（CatchAndDetect_local），当检测到ROS节点将图像写到本地时，输入照片到缓冲区(ROSCatchImg.cpp)

3. 对照片进行镜头校正与裁切(ROSCatchImg.cpp)

4. 利用opencv stitcher类对图像进行拼接（其实尝试过自己写stiticher类，但是效果不如opencv的）

5. 将拼接好的图像分为1180x640大小，输入yolo网络中（每个图像中有一定的重叠防止目标出现在接缝处）(imageCut.cpp)

6. 将裁切好的图像进行圆的检测（分为HSV提取红色区域和圆hough检测两部分）(circleProcess.cpp)

7. 由于已知圆的半径为1米，根据Hough检测得到的圆计算目标与中心的相对距离(darknetProcess.cpp)

8. 将boundingbox打印在大图上输出

<!--
## 存在的问题与改进
实际上在实机测试和赛场测试中遇到了很多问题，解释如下：
1. 为什么不把检测识别算法写入ROS中：
答：1.发现ROS自身集成的YOLO无法将检测结果的图像坐标输出，这样就无法对拼接好的图像运行yolo网络（yolo是onestage的网络，要求测试图像大小和训练图像大小一致，但是拼接图像的大小不一定，无法训练；如果将拼接图像裁切成小图的话最终打印boundingbox必须要求检测结果的图像坐标）。2.本来想通过opencv视频流打开图传图像链路，通过qgroundcontrol给于的指令（通过txt文件实现）拍摄若干照片，利用px4相机触发中的MAV_IMAGE_CAPTURED消息，但是发现mission模式中该方法并不可用
2. 为什么使用opencv的stitcher类
答：最开始尝试自己写stitcher类，因为希望能够导出stitcher类中的变换矩阵，但是发现效果不如opencv自身封装的stitcher类，甚至用opencv的stitcher-detail方法效果也不好，因此最终还是用了stitcher类。
但是这又出现了一个问题，stitcher类无法调整特征点匹配环节中的特征点数量，因为实际比赛场景中拍摄图像中特征点很少（草地，特征点数量远远不如北苑操场）。因此最终将采集到的图像先做直方图均衡预处理人为添加特征值(imagePreProcess.cpp)，在进行stitcher操作，但是效果仍然时好时坏。
3. 圆检测不准确
答：基于Hough变换的圆检测收到调参的影响，出现了检测不准确的问题，之后最好还是将圆也加入yolo中。
4. 实际比赛中的特殊待遇
答：实际比赛中，由于stitcher类不稳定，因此采取了1.对每张图像单张运行yolo检测并输出结果(ROSCatchImg.cpp)2.当图像采集高度和采集点间距一定的条件下，手动测量两张图像之间的重叠像素值，强行拼接（imageForceStitch.cpp)
5. 硬件问题
答：1.最开始使用的飞机和飞控不稳，造成大量炸机，耽误进度 2.图传和相机相对分辨率和成像效果不是很好，对图像拼接和识别都造成了一定影响 3.把amimon connex图传炸掉导致最后对insight图传的测试还是存在一些仓促  4.ROS对到达GPS坐标点的检测依赖P400数传，但是P400数传存在波特率和带宽不稳的状况，造成飞机实时GPS坐标可能不能准确传输，导致图像跳过。
6. 可以尝试的改进
答：1.发现有使用成品图像拼接软件的（空工大第一名），在不限制出结果时间的条件下是一个可选方案 2.降落到H点坐标实在是没时间写了，现在的人数配置搞整个比赛项目的硬件+软件+飞行+测试还是力不从心。3.最后的成绩也存在侥幸的因素在里面，整个程序还是有很多漏洞，比如必须要采集够20张图片才能进入之后的stitch，yolo环节；ROS中必须采集了第n张图才能采集第n+1张图，跳过一张后面的都才不了等。4.完全基于图像特征点的拼接存在较低的鲁棒性，对实际场景、图像重叠度等要求很高，也许之后可以考虑考虑vio
-->

cmake_minimum_required(VERSION 3.14)
project(CatchAndDetect)

set(CMAKE_CXX_STANDARD 14)
# 配置Opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

#导入头文件
include_directories(include)
add_executable(CatchAndDetect main.cpp darknetProcess.h darknetProcess.cpp imageProcess.h imageProcess.cpp imageCut.cpp imageCut.h circleProcess.h circleProcess.cpp insightCatchImg.cpp insightCatchImg.h amimonLocalImg.cpp amimonLocalImg.h insightLocalImg.cpp insightLocalImg.h)
#link_directories(/home/zhouhelu/darknet/libdarknet.so)
#target_link_libraries(CatchAndDetect ${OpenCV_LIBS} /home/zhouhelu/darknet/libdarknet.so)
target_link_libraries(CatchAndDetect ${OpenCV_LIBS})

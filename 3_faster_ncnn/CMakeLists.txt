cmake_minimum_required(VERSION 3.0)

project(yolo_ncnn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

include_directories(/home/zzx/Github/zzx_yolo/EXTRA_PKG/ncnn/build/install/include/ncnn)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")

add_executable(yolox_ncnn 
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/apps/common.cpp
    ${PROJECT_SOURCE_DIR}/src/apps/yolo/yolo.cpp
)
target_link_libraries(yolox_ncnn ${OpenCV_LIBS} ncnn pthread)

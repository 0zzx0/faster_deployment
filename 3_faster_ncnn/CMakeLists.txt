cmake_minimum_required(VERSION 3.0)

project(yolo_ncnn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/home/zzx/Github/zzx_yolo/EXTRA_PKG/ncnn/build/install/include/ncnn)


file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)


add_executable(yolox_ncnn ${cpp_srcs})
target_link_libraries(yolox_ncnn ${OpenCV_LIBS} ncnn pthread)

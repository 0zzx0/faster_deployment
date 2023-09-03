cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

project(trt_cpp LANGUAGES CXX CUDA)

# tensorrt
include_directories(/home/zzx/Github/zzx_yolo/EXTRA_PKG/TensorRT-8.5.3.1/include)
link_directories(/home/zzx/Github/zzx_yolo/EXTRA_PKG/TensorRT-8.5.3.1/lib)
 
# add CUDA
find_package(CUDA REQUIRED)
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})


SET(LIBHELLO_SRC demo01.cu demo01.h cookbookHelper.cuh)
ADD_LIBRARY(demo01 SHARED ${LIBHELLO_SRC})

# link
target_link_libraries(demo01 nvinfer ${CUDA_LIBRARIES})
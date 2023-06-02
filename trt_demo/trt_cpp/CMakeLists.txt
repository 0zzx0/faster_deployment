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
 
add_executable(main main.cpp cookbookHelper.cuh)
# link
# target_link_libraries(main ${LIBS} ${CUDA_LIBRARIES})
target_link_libraries(main nvinfer nvonnxparser ${CUDA_LIBRARIES})

cmake_minimum_required(VERSION 3.0)

project(yolo_trt)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# cudnn
include_directories(/home/zzx/Github/zzx_yolo/EXTRA_PKG/cudnn8.7.0.84_cuda10/include)
link_directories(/home/zzx/Github/zzx_yolo/EXTRA_PKG/cudnn8.7.0.84_cuda10/lib)
# tensorrt
include_directories(/home/zzx/Github/zzx_yolo/EXTRA_PKG/TensorRT-8.5.3.1/include)
link_directories(/home/zzx/Github/zzx_yolo/EXTRA_PKG/TensorRT-8.5.3.1/lib)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")

file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/kernels/*.cu)

cuda_add_library(cuda_kernels SHARED ${cuda_srcs})
target_link_libraries(cuda_kernels cuda cudart)
target_link_libraries(cuda_kernels ${OpenCV_LIBS})

add_executable(yolo 
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/base/infer_base.cpp
    ${PROJECT_SOURCE_DIR}/src/base/memory_tensor.cpp
    ${PROJECT_SOURCE_DIR}/src/base/trt_base.cpp
    ${PROJECT_SOURCE_DIR}/src/apps/yolo/yolo.cpp
    ${PROJECT_SOURCE_DIR}/src/apps/rtdetr/rtdetr.cpp
)
target_link_libraries(yolo cuda_kernels)
target_link_libraries(yolo nvinfer nvinfer_plugin nvonnxparser)
target_link_libraries(yolo cuda cublas cudart cudnn)
target_link_libraries(yolo pthread)
target_link_libraries(yolo ${OpenCV_LIBS})


add_executable(eval 
    ${PROJECT_SOURCE_DIR}/src/eval/eval.cpp
    ${PROJECT_SOURCE_DIR}/src/eval/save.hpp
    ${PROJECT_SOURCE_DIR}/src/base/infer_base.cpp
    ${PROJECT_SOURCE_DIR}/src/base/memory_tensor.cpp
    ${PROJECT_SOURCE_DIR}/src/base/trt_base.cpp
    ${PROJECT_SOURCE_DIR}/src/apps/yolo/yolo.cpp
    )
target_link_libraries(eval cuda_kernels)
target_link_libraries(eval nvinfer nvinfer_plugin nvonnxparser)
target_link_libraries(eval cuda cublas cudart cudnn)
target_link_libraries(eval pthread)
target_link_libraries(eval ${OpenCV_LIBS})

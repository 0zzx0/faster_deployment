cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(trt_cpp LANGUAGES CXX CUDA)
add_executable(main main.cpp cookbookHelper.cuh)

# tensorrt
include_directories(/tensorrt/TensorRT-8.6.0.12/include)
file(GLOB LIBS "/tensorrt/TensorRT-8.6.0.12/lib/*.so")
 
# add CUDA
find_package(CUDA 11.4 REQUIRED)
message("CUDA_LIBRARIES:${CUDA_LIBRARIES}")
message("CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
 
# link
target_link_libraries(main ${LIBS} ${CUDA_LIBRARIES})



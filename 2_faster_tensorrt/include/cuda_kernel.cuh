#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH
/*
定义一些自定义的CUDA操作，主要是预处理部分和后处理部分的加速
*/

#include <cuda_runtime.h>

#include "tools.hpp"

namespace YOLO{


#define GPU_BLOCK_THREADS  512      // gpu 每个block线程数量
const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag

static dim3 grid_dims(int numJobs);
static dim3 block_dims(int numJobs);

// sigmoid 和 逆sigmoid 具体是否使用看模型里面输出前有没有sigmoid吧
static __host__ inline float desigmoid(float y){
    return -log(1.0f/y - 1.0f);
}

static __device__ inline float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}


//////////////////////Norm part/////////////////
enum class NormType : int{
    None      = 0,
    MeanStd   = 1,
    AlphaBeta = 2
};

enum class ChannelType : int{
    None          = 0,
    SwapRB        = 1
};

/* 归一化操作，可以支持均值标准差，alpha beta，和swap RB */
struct Norm{
    float mean[3];
    float std[3];
    float alpha, beta;
    NormType type = NormType::None;
    ChannelType channel_type = ChannelType::None;

    // out = (x * alpha - mean) / std
    static Norm mean_std(const float mean[3], const float std[3], float alpha = 1/255.0f, ChannelType channel_type=ChannelType::None);
    
    // out = x * alpha + beta
    static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type=ChannelType::None);
    
    // None
    static Norm None();
};


static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy);
static __global__ void yolox_decode_kernel(
    float* predict, 
    int num_bboxes, 
    int fm_area,
    int num_classes, 
    float confidence_threshold, 
    float* invert_affine_matrix, 
    float* parray, 
    const float* prior_box,
    int max_objects
);
static __global__ void decode_kernel(
    float* predict, 
    int num_bboxes, 
    int fm_area,
    int num_classes, 
    float confidence_threshold,
    float* invert_affine_matrix, 
    float* parray, 
    const float* prior_box,
    int max_objects
);

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
);
static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold);

void yolox_decode_kernel_invoker(
    float* predict, 
    int num_bboxes, 
    int fm_area,
    int num_classes, 
    float confidence_threshold, 
    float nms_threshold, 
    float* invert_affine_matrix, 
    float* parray, 
    const float* prior_box,
    int max_objects, 
    cudaStream_t stream
);

void yolov5_decode_kernel_invoker(
    float* predict, 
    int num_bboxes, 
    int fm_area,
    int num_classes, 
    float confidence_threshold, 
    float nms_threshold, 
    float* invert_affine_matrix, 
    float* parray, 
    const float* prior_box,
    int max_objects, 
    cudaStream_t stream
);

static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
    uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm, int edge);
void warp_affine_bilinear_and_normalize_plane(
    uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
    float* matrix_2_3, uint8_t const_value, const Norm& norm,
    cudaStream_t stream);


#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)
template<typename _T>
static __inline__ __device__ _T limit(_T value, _T low, _T high){
    return value < low ? low : (value > high ? high : value);
}
static __inline__ __device__ int resize_cast(int value){
    return (value + (1 << (CAST_BITS - 1))) >> CAST_BITS;
}

static __global__ void resize_bilinear_and_normalize_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
    float sx, float sy, Norm norm, int edge
    );

void resize_bilinear_and_normalize(
    uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
    const Norm& norm,
    cudaStream_t stream
    );


};

#endif

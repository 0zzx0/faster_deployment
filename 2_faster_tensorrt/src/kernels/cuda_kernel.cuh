#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH
/*
定义一些自定义的CUDA操作，主要是预处理部分和后处理部分的加速
*/

#include <cuda_runtime.h>

#include "../base/tools.hpp"

namespace FasterTRT{


#define GPU_BLOCK_THREADS  512      // gpu 每个block线程数量
const int NUM_BOX_ELEMENT = 7;      // left, top, right, bottom, confidence, class, keepflag

// 用于插值计算的常量和函数
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

// sigmoid 和 逆sigmoid 具体是否使用看模型里面输出前有没有sigmoid吧
static __host__ inline float desigmoid(float y){
    return -log(1.0f/y - 1.0f);
}

static __device__ inline float sigmoid(float x){
    return 1.0f / (1.0f + exp(-x));
}


static dim3 grid_dims(int numJobs);
static dim3 block_dims(int numJobs);

//////////////////////归一化策略/////////////////
enum class NormType : int{
    None      = 0,
    MeanStd   = 1,
    AlphaBeta = 2
};

//////////////////////通道策略/////////////////
enum class ChannelType : int{
    None          = 0,
    SwapRB        = 1
};

/* 归一化操作，可以支持均值标准差，alpha beta 以及输入图片通道部分swap RB */
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

// 仿射变换
static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy);


// 计算iou
static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
);
// nms kernel
static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold);


// yolox的解码kernel
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

// yolox的解码
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

/**
 * @brief 通过仿射变换完成双线性插值resize并且归一化的kernel
 * 
 * @param src 原始图像数据
 * @param src_line_size 图像长度(宽度*3)
 * @param src_width 原始图像宽
 * @param src_height 原始图像高
 * @param dst 目标图像数据
 * @param dst_width 目标图像宽
 * @param dst_height 目标图像高
 * @param const_value_st padding值
 * @param warp_affine_matrix_2_3 仿射变化矩阵
 * @param norm 归一化策略
 * @param edge 目标图像范围
 */
static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
    uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm, int edge);

/**
 * @brief 通过仿射变换完成双线性插值resize并且归一化
 * 
 * @param src 原始图像数据
 * @param src_line_size 图像长度(宽度*3)
 * @param src_width 原始图像宽
 * @param src_height 原始图像高
 * @param dst 目标图像数据
 * @param dst_width 目标图像宽
 * @param dst_height 目标图像高
 * @param matrix_2_3 仿射变化矩阵
 * @param const_value padding值
 * @param norm 归一化策略
 * @param stream cuda stream
 */
void warp_affine_bilinear_and_normalize_plane(
    uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
    float* matrix_2_3, uint8_t const_value, const Norm& norm,
    cudaStream_t stream);



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

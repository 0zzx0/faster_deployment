#include "cuda_kernel.cuh"

/*
cuda kernel
*/

namespace FasterTRT{

// 确定gpu grid维度
static dim3 grid_dims(int numJobs) {
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

// 确定gpu block维度 尽量每个jobs一个thread
static dim3 block_dims(int numJobs) {
    return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

///////////////////////////////////////////////////////////////////////
//////////////////////////// Norm ////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// 均值方差归一化 
Norm Norm::mean_std(const float mean[3], const float std[3], float alpha, ChannelType channel_type){

    Norm out;
    out.type  = NormType::MeanStd;
    out.alpha = alpha;
    out.channel_type = channel_type;
    memcpy(out.mean, mean, sizeof(out.mean));
    memcpy(out.std,  std,  sizeof(out.std));
    return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type){

    Norm out;
    out.type = NormType::AlphaBeta;
    out.alpha = alpha;
    out.beta = beta;
    out.channel_type = channel_type;
    return out;
}

Norm Norm::None(){
    return Norm();
}	

// 仿射变换
// static __device__ inline void affine_project(float* matrix, float x, float y, float* ox, float* oy){
//     *ox = matrix[0] * x + matrix[1];
//     *oy = matrix[0] * y + matrix[2];
// }
static __device__ inline void affine_project(float* matrix, float x, float y, float* ox, float* oy){
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}


// 计算iou
static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
){

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}


// nms
static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;
    
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
} 


// yolox解码输出 
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
){  
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    // prior_box is 8400(axhxw) x 3
    // 1 x 8400 x 85
    float* pitem     = predict + position * fm_area;
    float objectness = pitem[4];
    if(objectness < confidence_threshold)
        return;

    // 找到置信度最高的class
    float confidence        = pitem[5];
    int label               = 0;
    for(int i = 1; i < num_classes; ++i){
        float class_confidence = pitem[i + 5];
        if(class_confidence > confidence){
            confidence = class_confidence;
            label      = i;
        } 
    }
    confidence *= objectness;
    if(confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;

    float predict_cx = pitem[0];
    float predict_cy = pitem[1];
    float predict_w  = exp(pitem[2]);
    float predict_h  = exp(pitem[3]);
    
    const float* prior_ptr = prior_box + position * 3;
    float stride     = prior_ptr[2];
    float cx         = (predict_cx + prior_ptr[0]) * stride;
    float cy         = (predict_cy + prior_ptr[1]) * stride;
    float width      = predict_w * stride;
    float height     = predict_h * stride;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}


// yolox 后处理
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
){
    auto grid = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkCudaKernel(yolox_decode_kernel<<<grid, block, 0, stream>>>(
        predict, 
        num_bboxes, 
        fm_area,
        num_classes, 
        confidence_threshold,
        invert_affine_matrix, 
        parray, 
        prior_box,
        max_objects
    ));

    grid = grid_dims(max_objects);
    block = block_dims(max_objects);
    checkCudaKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold));
}

// same to opencv
// reference: https://github.com/opencv/opencv/blob/24fcb7f8131f707717a9f1871b17d95e7cf519ee/modules/imgproc/src/resize.cpp
// reference: https://github.com/openppl-public/ppl.cv/blob/04ef4ca48262601b99f1bb918dcd005311f331da/src/ppl/cv/cuda/resize.cu


__global__ void resize_bilinear_and_normalize_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_width, int dst_height, 
    float sx, float sy, Norm norm, int edge
){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    int dx      = position % dst_width;
    int dy      = position / dst_width;
    float src_x = (dx + 0.5f) * sx - 0.5f;
    float src_y = (dy + 0.5f) * sy - 0.5f;
    float c0, c1, c2;

    int y_low = floorf(src_y);
    int x_low = floorf(src_x);
    int y_high = limit(y_low + 1, 0, src_height - 1);
    int x_high = limit(x_low + 1, 0, src_width - 1);
    y_low = limit(y_low, 0, src_height - 1);
    x_low = limit(x_low, 0, src_width - 1);

    int ly    = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
    int lx    = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
    int hy    = INTER_RESIZE_COEF_SCALE - ly;
    int hx    = INTER_RESIZE_COEF_SCALE - lx;
    int w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
    float* pdst = dst + dy * dst_width + dx * 3;
    uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
    uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
    uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
    uint8_t* v4 = src + y_high * src_line_size + x_high * 3;

    c0 = resize_cast(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]);
    c1 = resize_cast(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]);
    c2 = resize_cast(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]);

    // if(norm.channel_type == ChannelType::Invert){
    //     float t = c2;
    //     c2 = c0;  c0 = t;
    // }

    if(norm.type == NormType::MeanStd){
        c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
        c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
        c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
    }else if(norm.type == NormType::AlphaBeta){
        c0 = c0 * norm.alpha + norm.beta;
        c1 = c1 * norm.alpha + norm.beta;
        c2 = c2 * norm.alpha + norm.beta;
    }

    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}


void resize_bilinear_and_normalize(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    float* dst, int dst_width, int dst_height,
    const Norm& norm, cudaStream_t stream
    ){
    
    int jobs   = dst_width * dst_height;
    auto grid  = grid_dims(jobs);
    auto block = block_dims(jobs);
    
    checkCudaKernel(resize_bilinear_and_normalize_kernel << <grid, block, 0, stream >> > (
        src, src_line_size, src_width, src_height, 
        dst, dst_width, dst_height, 
        src_width/(float)dst_width, src_height/(float)dst_height, 
        norm, jobs
    ));
}


static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height, 
    uint8_t const_value_st, float* warp_affine_matrix_2_3, Norm norm, int edge){

    int position = blockDim.x * blockIdx.x + threadIdx.x;   // 每个thread处理当前位置三通道像素点
    if (position >= edge) return;

    // TODO: 考虑shared memory
    float m_x1 = warp_affine_matrix_2_3[0];
    float m_y1 = warp_affine_matrix_2_3[1];
    float m_z1 = warp_affine_matrix_2_3[2];
    float m_x2 = warp_affine_matrix_2_3[3];
    float m_y2 = warp_affine_matrix_2_3[4];
    float m_z2 = warp_affine_matrix_2_3[5];

    int dx      = position % dst_width;     // 计算目标图像的x坐标
    int dy      = position / dst_width;     // 计算目标图像的y坐标
    float src_x = m_x1 * dx + m_y1 * dy + m_z1; // 根据仿射变换计算目标图像对应原图上的坐标
    float src_y = m_x2 * dx + m_y2 * dy + m_z2; //
    float c0, c1, c2;

    if(src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height){
        // 越界使用padding值
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }else{
        // 计算双线性插值的像素点
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        float ly    = src_y - y_low;
        float lx    = src_x - x_low;
        float hy    = 1 - ly;
        float hx    = 1 - lx;
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        // 获取周围的像素的值
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;
        // 避免越界
        if(y_low >= 0){
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        
        if(y_high < src_height){
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        // same to opencv
        // 根据插值权重和周围像素的值计算新的像素值
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    if(norm.channel_type == ChannelType::SwapRB){
        float t = c2;
        c2 = c0;  c0 = t;
    }

    if(norm.type == NormType::MeanStd){
        c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
        c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
        c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
    }else if(norm.type == NormType::AlphaBeta){
        c0 = c0 * norm.alpha + norm.beta;
        c1 = c1 * norm.alpha + norm.beta;
        c2 = c2 * norm.alpha + norm.beta;
    }

    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void warp_affine_bilinear_and_normalize_plane(
    uint8_t* src, int src_line_size, int src_width, int src_height, float* dst, int dst_width, int dst_height,
    float* matrix_2_3, uint8_t const_value, const Norm& norm,
    cudaStream_t stream) {
    
    int jobs   = dst_width * dst_height;
    auto grid  = grid_dims(jobs);
    auto block = block_dims(jobs);
    
    checkCudaKernel(warp_affine_bilinear_and_normalize_plane_kernel <<<grid, block, 0, stream >>> (
        src, src_line_size,
        src_width, src_height, dst,
        dst_width, dst_height, const_value, matrix_2_3, norm, jobs
    ));
}



};


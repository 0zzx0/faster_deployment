/**
 * @file common.h
 * @author 0zzx0
 * @brief
 * @version 0.1
 * @date 2023-08-21
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef COMMON_H
#define COMMON_H

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "../base/tools.hpp"
#include "../base/trt_base.hpp"
#include "../base/infer_base.hpp"
#include "../base/memory_tensor.hpp"
#include "../base/monopoly_accocator.hpp"

namespace FasterTRT {

// 推理结果格式
struct Box {
    float left, top, right, bottom, confidence;
    int class_label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left),
          top(top),
          right(right),
          bottom(bottom),
          confidence(confidence),
          class_label(class_label) {}
};
typedef std::vector<Box> BoxArray;

// 仿射变换矩阵
struct AffineMatrix {
    float i2d[6];  // image to dst(network), 2x3 matrix
    float d2i[6];  // dst to image, 2x3 matrix

    void compute(const cv::Size &from, const cv::Size &to) {
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;
        float scale = std::min(scale_x, scale_y);
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = -scale * from.width * 0.5 + to.width * 0.5 + scale * 0.5 - 0.5;
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);
    }

    cv::Mat i2d_mat() { return cv::Mat(2, 3, CV_32F, i2d); }
};

}  // namespace FasterTRT

#endif
/**
 * @file yolo.h
 * @author 0zzx0
 * @brief 重写改进，集成高性能yolo推理接口
 * @version 1.0
 * @date 2023-6-11
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef YOLO_HPP
#define YOLO_HPP

#include <future>
#include <memory>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "../../base/tools.hpp"
#include "../../base/trt_base.hpp"
#include "../../base/infer_base.hpp"
#include "../../base/memory_tensor.hpp"
#include "../../base/monopoly_accocator.hpp"

namespace YOLO {
using namespace FasterTRT;

// 模型选择
enum class YoloType : int { V5 = 0, X = 1, V8 = 2 };

// 模型名字
const char *type_name(YoloType type);

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

/**
 * @brief Decode配置的实现
 *  不同模型输出的decode一般都不一样，即使是yolo系列也有一些区别，
 *  这里需要实现不同模型的decode。尤其是anchor base 和anchor free的区别
 */
struct DecodeMeta {
    int num_anchor;
    int num_level;
    float w[16], h[16];
    int strides[16];

    // static DecodeMeta v5_p5_default_meta();
    static DecodeMeta x_default_meta();
    static DecodeMeta v8_default_meta();
};

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

// 线程安全模板类设置模板类型
using ThreadSafedAsyncInferImpl =
    ThreadSafedAsyncInfer<cv::Mat,                       // input
                          BoxArray,                      // output
                          std::tuple<std::string, int>,  // start param
                          AffineMatrix                   // additional
                          >;
using Infer = InferBase<cv::Mat, BoxArray>;

/**
 * @brief 推理类的实现，继承必备父类，重写父类方法
 *
 */
class YoloTRTInferImpl : public Infer, public ThreadSafedAsyncInferImpl {
public:
    // 析构 调用来自基类ThreadSafedAsyncInferImpl的stop函数
    ~YoloTRTInferImpl();

    virtual bool startup(const std::string &file, YoloType type, int gpuid, int batch_size,
                         float confidence_threshold, float nms_threshold,
                         bool is_use_trtnNMSPlugin = false);
    virtual void worker(std::promise<bool> &result) override;
    virtual bool preprocess(Job &job, const cv::Mat &image) override;

    virtual std::vector<std::shared_future<BoxArray>> commits(
        const std::vector<cv::Mat> &images) override;
    virtual std::shared_future<BoxArray> commit(const cv::Mat &image) override;

    void init_yolox_prior_box(Tensor &prior_box);
    void init_yolov8_prior_box(Tensor &prior_box);
    void init_yolov5_prior_box(Tensor &prior_box);

private:
    int input_width_ = 0;
    int input_height_ = 0;
    int gpu_ = 0;
    float confidence_threshold_ = 0;
    float nms_threshold_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaStream_t stream_pro_ = nullptr;
    Norm normalize_;
    YoloType type_;
    DecodeMeta meta_;
    int batch_size_ = 1;
    bool is_use_trtnNMSPlugin_ = false;
};

/*
    trt模型编译(不过我实际建议直接用trtexec转换,嘻嘻0_0)
    max max_batch_size：为最大可以允许的batch数量
    source_onnx_file：onnx文件
    save_engine_file：储存的tensorRT模型
    max_workspace_size：最大工作空间大小，一般给1GB，在嵌入式可以改为256MB，单位是byte
    int8 images
   folder：对于Mode为INT8时，需要提供图像数据进行标定，请提供文件夹，会自动检索下面的jpg/jpeg/tiff/png/bmp
    int8_entropy_calibrator_cache_file：对于int8模式下，熵文件可以缓存，避免二次加载数据，可以跨平台使用，是一个txt文件
*/
bool compile(Mode mode, YoloType type, unsigned int max_batch_size,
             const std::string &source_onnx_file, const std::string &save_engine_file,
             size_t max_workspace_size = 1 << 30, const std::string &int8_images_folder = "",
             const std::string &int8_entropy_calibrator_cache_file = "");

// image转成tensor
void image_to_tensor(const cv::Mat &image, std::shared_ptr<Tensor> &tensor, YoloType type,
                     int ibatch);

// 创建推理器
std::shared_ptr<Infer> create_infer(const std::string &engine_file, YoloType type, int gpuid,
                                    int batch_size, float confidence_threshold = 0.2f,
                                    float nms_threshold = 0.5f);

};  // end namespace YOLO

#endif
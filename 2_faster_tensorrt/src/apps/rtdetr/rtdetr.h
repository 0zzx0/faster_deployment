/**
 * @file rtdetr.h
 * @author 0zzx0
 * @brief RTDETR推理
 * @version 0.1
 * @date 2023-08-21
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef RTDETR_H
#define RTDETR_H

#include "../common.hpp"

namespace RTDETR {
using namespace FasterTRT;

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
class RtDetrTRTInferImpl : public Infer, public ThreadSafedAsyncInferImpl {
public:
    ~RtDetrTRTInferImpl();

    virtual bool startup(const std::string &file, int gpuid, int batch_size,
                         float confidence_threshold);
    virtual void worker(std::promise<bool> &result) override;
    virtual bool preprocess(Job &job, const cv::Mat &image) override;

    virtual std::vector<std::shared_future<BoxArray>> commits(
        const std::vector<cv::Mat> &images) override;
    virtual std::shared_future<BoxArray> commit(const cv::Mat &image) override;

private:
    int input_width_ = 0;
    int input_height_ = 0;
    int gpu_ = 0;
    float confidence_threshold_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaStream_t stream_pro_ = nullptr;
    Norm normalize_;
    int batch_size_ = 1;
};

// 创建推理器
std::shared_ptr<Infer> create_infer(const std::string &engine_file, int gpuid, int batch_size,
                                    float confidence_threshold = 0.2f);

}  // namespace RTDETR

#endif
